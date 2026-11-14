// Regression check for the inline SVG widgets in src/content/docs.
//
// Two failure modes are invisible in a normal build and both have bitten this repo:
//   1. A blank line inside a raw <script> block ends the HTML block in CommonMark, so
//      the rest of the script is parsed as Markdown — smartypants rewrites 'id' to
//      curly quotes and a*b*c becomes emphasis. The build still succeeds.
//   2. Slider ranges that drive drawn coordinates outside the SVG's viewBox, so parts
//      of the picture silently vanish at the extremes.
//
// This runs each widget's script against a minimal DOM shim, sweeps every slider to
// its min/mid/max in all combinations, and checks both. Run after `astro build`:
//   node scripts/check-widgets.mjs [distDir]
import fs from 'node:fs'
import path from 'node:path'

const DIST = process.argv[2] || 'dist'
const XA = new Set(['x', 'cx', 'x1', 'x2'])
const YA = new Set(['y', 'cy', 'y1', 'y2'])
const HIDDEN = '-100' // the widgets' convention for parking an element off-canvas

function pages(dir) {
  const out = []
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) out.push(...pages(p))
    else if (e.name === 'index.html' && fs.readFileSync(p, 'utf8').includes('<figure class="widget"')) out.push(p)
  }
  return out
}

function shim(html) {
  const all = []
  const mk = (tag, attrs = {}) => {
    const n = {
      tag, attrs: { ...attrs }, children: [], listeners: {}, bad: [],
      get firstChild() { return n.children[0] || null },
      get value() { return n.attrs.value },
      set value(v) { n.attrs.value = String(v) },
      setAttribute(k, v) {
        if (v == null || (typeof v === 'number' && !Number.isFinite(v)) || /NaN|undefined|Infinity/.test(String(v)))
          all.bad.push(`${n.attrs.id || tag}.${k} = ${v}`)
        n.attrs[k] = String(v)
      },
      getAttribute(k) { return n.attrs[k] },
      appendChild(c) { c.dead = false; n.children.push(c); return c },
      removeChild(c) { c.dead = true; n.children = n.children.filter((x) => x !== c); return c },
      addEventListener(ev, fn) { (n.listeners[ev] ||= []).push(fn) },
      set textContent(v) { if (v == null || /NaN|undefined/.test(String(v))) all.bad.push(`${n.attrs.id || tag}.text = ${v}`) },
      get textContent() { return '' },
      querySelector: () => null,
    }
    all.push(n)
    return n
  }
  all.bad = []
  const byId = new Map()
  for (const m of html.matchAll(/<(\w[\w-]*)\s([^>]*?)\/?>/g)) {
    const attrs = {}
    for (const a of m[2].matchAll(/([\w:-]+)="([^"]*)"/g)) attrs[a[1]] = a[2]
    if (attrs.id) byId.set(attrs.id, mk(m[1], attrs))
  }
  return { all, byId, document: { getElementById: (i) => byId.get(i) || null, createElementNS: (_n, t) => mk(t), querySelector: () => null, addEventListener() {} } }
}

function outside(all, viewBox) {
  const [vx, vy, vw, vh] = viewBox
  for (const n of all) {
    if (n.tag === 'svg' || n.dead || n.attrs.x === HIDDEN || n.attrs.x1 === HIDDEN) continue
    const pts = []
    for (const [k, v] of Object.entries(n.attrs)) {
      const f = parseFloat(v)
      if (!Number.isFinite(f)) continue
      if (XA.has(k)) pts.push([f, null])
      else if (YA.has(k)) pts.push([null, f])
    }
    for (const p of (n.attrs.points || '').trim().split(/\s+/).filter(Boolean)) {
      const [a, b] = p.split(',').map(Number)
      if (Number.isFinite(a)) pts.push([a, b])
    }
    for (const [x, y] of pts) {
      const ox = x != null && (x < vx - 1 || x > vx + vw + 1)
      const oy = y != null && Number.isFinite(y) && (y < vy - 1 || y > vy + vh + 1)
      if (ox || oy) return `"${n.attrs.id || n.tag}" leaves the viewBox at (${x ?? '-'}, ${y ?? '-'})`
    }
  }
  return null
}

let failed = 0
for (const file of pages(DIST)) {
  const name = path.relative(DIST, path.dirname(file))
  const html = fs.readFileSync(file, 'utf8')
  const fig = html.slice(html.indexOf('<figure class="widget"'))
  const staticVB = fig.match(/viewBox="([-\d.\s]+)"/)[1].trim().split(/\s+/).map(Number)
  const body = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).find((b) => b.includes("getElementById('w-"))

  if (!body) { console.log(`✗ ${name}: widget markup present but no widget script`); failed++; continue }
  if (/[‘’“”]/.test(body) || /<\/?em>|<\/?strong>/.test(body)) {
    console.log(`✗ ${name}: script was mangled by Markdown (curly quotes or emphasis tags) — check for a blank line inside <script>`)
    failed++
    continue
  }

  const { all, byId, document } = shim(html)
  try { new Function('document', 'window', body)(document, {}) }
  catch (e) { console.log(`✗ ${name}: threw on load — ${e.message}`); failed++; continue }

  const svgNode = [...byId.values()].find((n) => n.tag === 'svg' && n.attrs.viewBox)
  const inputs = [...byId.values()].filter((n) => n.attrs.type === 'range')
  let escape = null
  let runs = 0
  const combos = inputs.map((i) => { const lo = +i.attrs.min, hi = +i.attrs.max; return [lo, (lo + hi) / 2, hi] })
  const sweep = (d) => {
    if (escape) return
    if (d === inputs.length) {
      for (const fn of inputs[0].listeners.input || []) fn({})
      runs++
      escape ||= outside(all, svgNode ? svgNode.attrs.viewBox.trim().split(/\s+/).map(Number) : staticVB)
      return
    }
    for (const v of combos[d]) { inputs[d].value = v; sweep(d + 1) }
  }
  if (inputs.length) sweep(0)

  const problem = escape || (all.bad.length ? `${all.bad.length} non-finite values, e.g. ${all.bad[0]}` : null)
  if (problem) { console.log(`✗ ${name}: ${problem}`); failed++ }
  else console.log(`✓ ${name.padEnd(32)} ${inputs.length} sliders, ${runs} combinations`)
}
console.log(failed ? `\n${failed} widget(s) failing` : '\nall widgets execute cleanly and stay inside their viewBox')
process.exit(failed ? 1 : 0)
