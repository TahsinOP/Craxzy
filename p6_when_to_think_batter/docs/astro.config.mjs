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
      title: 'When-to-Think Batter',
      description: 'A Panda arm swings a 1.2 cm bat at pitched balls in MuJoCo 4.0. The planner is expensive; the question is when to run it.',
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
        { label: 'II · The Batting Cage', items: [{ autogenerate: { directory: 'simulation' } }] },
        { label: 'III · Seeing the Ball', items: [{ autogenerate: { directory: 'estimation' } }] },
        { label: 'IV · The Expensive Planner', items: [{ autogenerate: { directory: 'planner' } }] },
        { label: 'V · A Learned Planner', items: [{ autogenerate: { directory: 'world-model' } }] },
        { label: 'VI · When to Think', items: [{ autogenerate: { directory: 'when-to-think' } }] },
        { label: 'VII · Surprises and Events', items: [{ autogenerate: { directory: 'surprises' } }] },
        { label: 'VIII · Stretch: The Rally', items: [{ autogenerate: { directory: 'rally' } }] },
        { label: 'IX · Lessons and Limits', items: [{ autogenerate: { directory: 'lessons' } }] },
        { label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
      ],
    }),
  ],
})
