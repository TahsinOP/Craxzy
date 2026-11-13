// astro.config.mjs
import starlight from '@astrojs/starlight'
import { defineConfig } from 'astro/config'
import starlightThemeNova from 'starlight-theme-nova'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

// Set by `pnpm build:offline`: emits a dist-offline/ that opens straight from file://
const offline = process.env.OFFLINE === '1'

export default defineConfig({
  outDir: offline ? './dist-offline' : './dist',
  ...(offline && { prefetch: false }),
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      plugins: [starlightThemeNova()],
      // Pagefind fetches its index at runtime, which file:// blocks.
      pagefind: !offline,
      title: 'Card Catcher',
      description: 'A Panda arm catches a ball, a plate and a playing card in MuJoCo 4.0: physics, estimation, a learned residual, a QP interception controller and event-triggered replanning.',
      customCss: ['./src/styles/custom.css'],
      // Simulation clips play only while on screen. Inline and classic, so it also runs from file://.
      head: [
        {
          tag: 'script',
          content: `document.addEventListener('DOMContentLoaded',()=>{if(!('IntersectionObserver'in window))return;const io=new IntersectionObserver(es=>{for(const e of es){const v=e.target;if(e.isIntersecting){v.play().catch(()=>{})}else if(!v.paused){v.pause()}}},{threshold:0.35});document.querySelectorAll('figure.sim video').forEach(v=>io.observe(v))})`,
        },
      ],
      sidebar: [
        { label: 'I · The Problem and the Idea', items: [{ autogenerate: { directory: 'overview' } }] },
        { label: 'II · Simulation', items: [{ autogenerate: { directory: 'simulation' } }] },
        { label: 'III · Estimation', items: [{ autogenerate: { directory: 'estimation' } }] },
        { label: 'IV · The Learned Residual', items: [{ autogenerate: { directory: 'residual' } }] },
        { label: 'V · Interception Control', items: [{ autogenerate: { directory: 'control' } }] },
        { label: 'VI · When to Replan', items: [{ autogenerate: { directory: 'replanning' } }] },
        { label: 'VII · Evaluation', items: [{ autogenerate: { directory: 'evaluation' } }] },
        { label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
      ],
    }),
  ],
})
