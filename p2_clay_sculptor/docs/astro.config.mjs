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
			title: 'Clay Sculptor',
			description:
				'A Panda arm presses, rolls and stamps plastic clay into a target shape in MuJoCo 4.0: elasto-plastic flex, a geometric prior, learned dynamics, sampling-based MPC and an ensemble that knows when to ask the simulator.',
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
				{ label: 'II · The Clay Bench', items: [{ autogenerate: { directory: 'bench' } }] },
				{ label: 'III · Shape', items: [{ autogenerate: { directory: 'shape' } }] },
				{ label: 'IV · Planning with the Simulator', items: [{ autogenerate: { directory: 'planning' } }] },
				{ label: 'V · Learned Dynamics', items: [{ autogenerate: { directory: 'dynamics' } }] },
				{ label: 'VI · Planning with Learned Models', items: [{ autogenerate: { directory: 'learned-mpc' } }] },
				{ label: 'VII · Knowing When the Model Is Wrong', items: [{ autogenerate: { directory: 'fallback' } }] },
				{ label: 'VIII · Evaluation', items: [{ autogenerate: { directory: 'evaluation' } }] },
				{ label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
			],
		}),
	],
})
