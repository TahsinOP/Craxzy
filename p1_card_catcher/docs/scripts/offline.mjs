// Post-processes an Astro/Starlight build into something that works from file://
//   - rewrites every root-absolute URL (/foo/) into a relative one (../foo/index.html)
//   - re-bundles module scripts as classic scripts, since Chrome blocks ESM over file://
// Run via `pnpm build:offline`; see README-offline.md.
import { createRequire } from 'node:module'
import fs from 'node:fs/promises'
import path from 'node:path'

const ROOT = path.resolve('dist-offline')
const SKIP = /^(?:[a-z][a-z0-9+.-]*:|\/\/|#|$)/i

async function walk(dir) {
  const out = []
  for (const e of await fs.readdir(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) out.push(...(await walk(p)))
    else out.push(p)
  }
  return out
}

/** Absolute site path -> path relative to the file doing the referencing. */
function relativize(url, fromDir) {
  const [, bare, tail = ''] = url.match(/^([^?#]*)([?#].*)?$/)
  let target = bare.replace(/^\//, '')
  const last = target.split('/').pop()
  if (target === '') target = 'index.html'
  else if (bare.endsWith('/')) target += 'index.html'
  else if (!last.includes('.')) target += '/index.html'
  const rel = path.relative(fromDir, path.join(ROOT, target)) || 'index.html'
  return rel + tail
}

async function bundleScripts(entries) {
  let esbuild
  try {
    const [dir] = (await fs.readdir(path.resolve('node_modules/.pnpm')))
      .filter((d) => d.startsWith('esbuild@'))
    esbuild = createRequire(import.meta.url)(
      path.resolve('node_modules/.pnpm', dir, 'node_modules/esbuild')
    )
  } catch {
    console.warn('! esbuild not found - skipping script bundling (nav/TOC will be inert)')
    return false
  }
  await esbuild.build({
    entryPoints: [...entries].map((e) => path.join(ROOT, e.replace(/^\//, ''))),
    outdir: path.join(ROOT, '_astro/offline'),
    bundle: true,
    format: 'iife',
    minify: true,
    target: 'es2020',
    logLevel: 'warning',
  })
  return true
}

const files = await walk(ROOT)
const html = files.filter((f) => f.endsWith('.html'))

// Pass 1: collect module-script entry points across every page.
const entries = new Set()
for (const f of html) {
  const src = await fs.readFile(f, 'utf8')
  for (const m of src.matchAll(/<script[^>]*type="module"[^>]*src="(\/[^"]+)"/g)) {
    if (!/\/Search\./.test(m[1])) entries.add(m[1]) // search is dead weight without pagefind
  }
}
const bundled = await bundleScripts(entries)

// Pass 2: rewrite the HTML.
for (const f of html) {
  const dir = path.dirname(f)
  let src = await fs.readFile(f, 'utf8')

  src = src.replace(/<link\b[^>]*rel="modulepreload"[^>]*>/g, '')
  src = src.replace(/<site-search\b[\s\S]*?<\/site-search>/g, '')
  src = src.replace(/<script[^>]*src="\/_astro\/Search\.[^"]*"[^>]*><\/script>/g, '')

  if (bundled) {
    src = src.replace(
      /<script([^>]*?)\stype="module"([^>]*?)\ssrc="(\/_astro\/[^"]+)"([^>]*)><\/script>/g,
      (_m, a, b, url, c) =>
        `<script${a}${b} defer src="${url.replace('/_astro/', '/_astro/offline/')}"${c}></script>`
    )
  }

  src = src.replace(/\b(href|src)="(\/[^"]*)"/g, (m, attr, url) =>
    SKIP.test(url.slice(1)) && !url.startsWith('/') ? m : `${attr}="${relativize(url, dir)}"`
  )
  src = src.replace(/\bsrcset="([^"]*)"/g, (m, val) => {
    if (!val.includes('/')) return m
    const out = val.split(',').map((part) => {
      const [u, ...rest] = part.trim().split(/\s+/)
      return u.startsWith('/') ? [relativize(u, dir), ...rest].join(' ') : part.trim()
    })
    return `srcset="${out.join(', ')}"`
  })
  src = src.replace(/url\((['"]?)(\/[^)'"]+)\1\)/g, (_m, q, url) => `url(${q}${relativize(url, dir)}${q})`)

  await fs.writeFile(f, src)
}

// Pass 3: rewrite asset URLs inside CSS (KaTeX fonts, mostly).
for (const f of files.filter((f) => f.endsWith('.css'))) {
  const dir = path.dirname(f)
  const src = await fs.readFile(f, 'utf8')
  const out = src.replace(/url\((['"]?)(\/[^)'"]+)\1\)/g, (_m, q, url) => `url(${q}${relativize(url, dir)}${q})`)
  if (out !== src) await fs.writeFile(f, out)
}

console.log(`✓ ${html.length} pages rewritten for file:// - open ${path.join(ROOT, 'index.html')}`)
