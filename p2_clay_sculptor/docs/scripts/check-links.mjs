// Every internal link and every /fig/ or /video/ source in the built site must resolve.
//   node scripts/check-links.mjs [distDir]
import fs from 'node:fs'
import path from 'node:path'

const DIST = process.argv[2] || 'dist'
const pages = []
const walk = (d) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); e.isDirectory() ? walk(p) : e.name === 'index.html' && pages.push(p) } }
walk(DIST)
let bad = 0
for (const file of pages) {
  const html = fs.readFileSync(file, 'utf8')
  const page = '/' + path.relative(DIST, path.dirname(file)).replace(/\\/g, '/')
  for (const m of html.matchAll(/(?:href|src)="(\/[^"#?]*)/g)) {
    const url = m[1]
    if (url.startsWith('/_astro/') || url.startsWith('/pagefind/')) continue
    const target = path.join(DIST, url)
    const ok = fs.existsSync(target) && (fs.statSync(target).isFile() || fs.existsSync(path.join(target, 'index.html')))
    if (!ok) { console.log(`✗ ${page} -> ${url}`); bad++ }
  }
  const katexErr = (html.match(/class="katex-error"/g) || []).length
  if (katexErr) { console.log(`✗ ${page}: ${katexErr} KaTeX error(s)`); bad++ }
}
console.log(bad ? `\n${bad} problem(s) in ${pages.length} pages` : `all links resolve in ${pages.length} pages, no KaTeX errors`)
process.exit(bad ? 1 : 0)
