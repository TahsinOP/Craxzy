---
title: "All Results"
description: "Every milestone in one page: what was measured, the headline number, what held up, what did not, and where the details are."
sidebar:
  order: 1
---

:::note[Guiding question]
The brief made three bets: that a geometric prior makes a learned clay model better, that the better model makes a better planner, and that an ensemble can tell the planner when to ask the simulator instead. Which of the three came true, on this bench?
:::

## In one table

| milestone | question | answer | details |
|---|---|---|---|
| M0 bench | Does the simulated clay keep a dent, replay exactly, and follow the arm? | Yes. A 10 mm press keeps 6.7 mm, and the dent does not creep (0.01 mm). Restoring the full state replays a rollout bit for bit; leaving the plastic strain out costs 3.4 mm. The press reaches 10.0 mm of a commanded 10.0 mm. | [The Clay](/bench/clay/), [State](/bench/state/) |
| M1 oracle | How good is the planner when its model is exact? | 13.9 ± 1.9 % of the initial error left, at 24 s and 96 simulations per decision. Zero was reachable; the greedy search is what stops short. | [The Oracle](/planning/oracle/) |
| M2 models | How well do the models predict one primitive? | No change 1.74 mm, prior 0.61, GNN 0.32, structured 0.27, U-Net 0.15 mm. The prior makes the graph network 17 % better; the U-Net is twice as good as both. | [How Good Are the Models?](/dynamics/results/) |
| M3 learned MPC | Do better predictions give better shapes? | Much less than they promise. Oracle 13.9 %, U-Net 14.6 %, prior 15.0 %, structured 15.8 %, GNN 17.3 % of the initial error left, at 30 to 400 times less planning time. | [Planning with Learned Models](/learned-mpc/results/) |
| M4 fallback | Can the ensemble say when to ask the simulator? | Not on this bench. The ensemble's spread predicts its own error well (Spearman 0.79), but as a trigger it is worse than asking the simulator at random (+1.1 points at the median threshold, p = 0.0004); a trigger on the members' doubt about the ranking ties with random. Asking does pay: re-ranking the best eight candidates at every decision gains 1 point, half of the oracle's advantage, at 1/12 of its simulations. | [Does the Fallback Pay?](/fallback/results/) |
| stretch | A letter, and a planner that picks the tool | The T: every planner at about 0.66 mm (14 %). Three tools: the learned planners reach 1.3 to 1.6 mm, the single oracle run 2.0 mm. Neither is solved; greed is the limit. | [A Letter, and a Choice of Tool](/evaluation/stretch/) |

## One demo: every planner on a target nobody tuned for

The targets above were all in use while the planners were built. As a last check, a new one was added after every model had been trained and every setting fixed: a **staircase** of three square pockets along the diagonal, pressed 6, 9 and 12 mm deep and turned 45°. No model was trained on any target (they learned from random single presses, rolls and stamps), and nothing was changed for this one. Three planner seeds each, five presses allowed:

| planner | height-map RMSE reached | initial error left | planning time per decision | simulated primitives per episode |
|---|---|---|---|---|
| U-Net | 0.40 ± 0.06 mm | 10.5 % | 0.06 s | 0 |
| hybrid: ensemble + MuJoCo re-ranks best 8 | 0.44 ± 0.06 mm | 11.5 % | 5.8 s | 40 |
| structured | 0.47 ± 0.03 mm | 12.3 % | 0.73 s | 0 |
| GNN | 0.47 ± 0.03 mm | 12.5 % | 0.78 s | 0 |
| prior only | 0.49 ± 0.05 mm | 13.1 % | 0.10 s | 0 |
| oracle (MuJoCo for every candidate) | 0.52 ± 0.17 mm | 13.9 % | 18.6 s | 480 |

Untouched block: 3.78 mm. With three seeds the order of this table means little: the spreads overlap, and the oracle's last place comes from one seed (0.34, 0.47 and 0.75 mm).

What the video shows is that every planner reads the staircase off the target map on its own. In the first seed, all six press the three pockets first and in order of size, each within about 10 mm of where the demonstration pressed, and then spend their last one or two presses on touch-ups. All six repeat the greedy pattern of the main targets: the deep pocket is pressed 12.5 to 14 mm (the demonstration used 12), the middle one about 7 mm (9) and the shallow one about 5 mm (6).

<figure class="sim">
<video src="/video/unseen.mp4" loop muted playsinline controls preload="metadata"></video>
<figcaption>All six planners on the unseen staircase, first seed, at half speed with the planning pauses cut. Each panel shows the arm and the clay's height map (yellow is the untouched top, blue is 12 mm down), with the live error to the target.</figcaption>
</figure>

### A smiley on a bigger block

A harder test changes the block as well as the target: a **smiley**, two eyes pressed 12 mm and a smile of five overlapping 10 mm presses on an arc, on a block of 245 mm instead of 140 (1.75 times the side, three times the area), with ten presses allowed. The cage keeps the cell size of training (21 × 21 × 3 cells), and the trained networks are reused without retraining: their weights do not depend on the size of the grid, only their reference buffers (node positions, edges, pixel coordinates), which are rebuilt for the bigger block. One planner seed:

| planner | height-map RMSE reached | initial error left | planning time per decision | simulated primitives |
|---|---|---|---|---|
| GNN | 0.88 mm | 35 % | 2.9 s | 0 |
| oracle | 0.89 mm | 36 % | 94 s | 960 |
| hybrid | 0.93 mm | 38 % | 24.8 s | 80 |
| prior only | 1.04 mm | 42 % | 0.7 s | 0 |
| structured | 1.10 mm | 44 % | 3.6 s | 0 |
| U-Net | 1.26 mm | 51 % | 0.2 s | 0 |

Untouched block: 2.48 mm. Every planner draws a recognisable face; none finishes it in ten presses, and the error left is about three times that of the small targets. The demonstration needs seven presses in exactly the right places, and a greedy planner spends some of its ten on partial overlaps. The graph networks, which see the clay as local neighbourhoods and the tool in its own frame, carry over to the bigger block; the GNN is level with the oracle here. The U-Net is last: it takes the absolute pixel position as an input, and on the bigger block most pixels lie outside the range it was trained on. One seed ranks nothing precisely; the ordering of the middle four is within what seeds change on the small targets.

<figure class="sim">
<video src="/video/smiley.mp4" loop muted playsinline controls preload="metadata"></video>
<figcaption>All six planners drawing a smiley on the 245 mm block, at half speed with the planning pauses cut. Right of each arm: the height map, seen from the camera's side.</figcaption>
</figure>

## The three bets

### 1. The prior makes a learned model better: yes, modestly

The structured model and the GNN are identical except for one input, the prior's displacement. With it, the one-step error falls from 0.323 to 0.267 mm, for every tool and at every horizon up to six chained steps, at twelve times the seed-to-seed spread. In the planner, the gain shrinks to about 1.5 percentage points of the initial error (paired over 30 episodes, p = 0.03).

What the prior does *not* do is make the model the best one. A U-Net on the height map, with no prior and no notion of the tool's shape, predicts twice as well. That comparison is lopsided (six times the parameters, trained to convergence, on the quantity it is scored by), and it is a statement about what a CPU trains in a quarter of an hour.

The prior on its own, with no learning, removes two thirds of the error of predicting nothing, from a single formula shared by the three tools.

### 2. A better model makes a better planner: only a little

A factor of two in one-step error became one or two percentage points of final error. The planner sees the true shape after every action, it needs the ranking of candidates rather than their values, and its own search error (13.9 % with an exact model) is larger than most of what the models add.

The sharpest lesson came from the slab. The prior alone, the worst predictor, planned the best slab of all, better than the simulator itself, because its error (no spring-back) made it ask for shallow strokes. Every accurate planner dug the first pass 12 to 13 mm deep and could not raise the clay again. For a greedy planner on a material that cannot be un-pressed, *which way* a model is wrong mattered more than *how much*.

### 3. The ensemble knows when to ask: no

The five-member ensemble passes the usual test of an uncertainty estimate: its spread ranks the true error of a prediction with a Spearman correlation of 0.79. As a trigger for the simulator, it failed the control experiment. At the same rate of simulator calls, firing when the members disagree left *more* error than firing at random, by 1.1 points at the median threshold (paired, p = 0.0004). A second trigger, built on the members' doubt about the *order* of the best candidates rather than about a shape, did exactly as well as random and no better.

The reasons were measurable. The spread grows with the size of the change being predicted, so it fired on almost every slab decision and on no dent decision, while the re-ranking gained as much per decision on the dent. And five networks of one architecture trained on one dataset share their errors: all five ranked the chosen candidate first at a third of the decisions, while the simulator found a better candidate among the eight at half of them.

What did work is the fallback itself, used as a judge: re-ranking the model's eight best candidates with the simulator gains one point of the initial error, half of what the exact planner gains, for 30 simulated primitives per episode against its 371. The bet that the model can say *when* it needs this was lost; the cheaper bet, that the simulator is worth asking about a short list, was won.

## What did not work, or was not done

- **The planner is greedy.** All targets were reachable with zero error and none was reached. A two-step search is the obvious next step, and the learned models are what would make it affordable.
- **The guided proposal ignores spring-back** and starts every search too shallow; all ten first stamps of the X were too shallow, even with the exact model. Found late and left in place so that all planners stay compared at the same search.
- **The height-map cost is blind to where the material is.** On the slab, every planner reaches 15 % of the initial height-map error while the earth mover's distance stays at 6 to 8 mm of the untouched block's 11.6.
- **The spring-back is numerical**, a property of the coarse cage that keeps the clay stable, not of the material law.
- **The graph networks are small and undertrained** next to the U-Net, for lack of a GPU.
- **Two stretch goals were not attempted**: folding the plastic paper sheet, and a second arm.

All of it is in [Limits](/evaluation/limits/).

## How the numbers were made

Every number on this site is written by a script into `results/*.json` and copied into the pages from there, or from `results/M0.md` to `M4.md`, which `scripts/make_reports.py` generates from the same files. Closed-loop comparisons use ten planner seeds per target (30 episodes per planner), paired by seed and target; prediction errors use three training seeds and 600 held-out transitions. [Running It](/reference/running-it/) lists the commands, and every figure is drawn from the same JSON by `docs/figures/make_figures.py`.
