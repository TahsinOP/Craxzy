---
title: "The Clay"
description: "An elasto-plastic solid in MuJoCo 4.0: the elastic law, the yield bound and the return mapping, the trilinear cage that carries it, and the measurements that decided between two discretisations."
sidebar:
  order: 1
---

:::note[Guiding question]
Plastic material in a rigid-body simulator is new. Which of its behaviours are the material law, which are the discretisation, and which discretisation can be pressed hard, released, and trusted to stay where it was left?
:::

## What plastic means

Push a rubber block and let go: it returns. Push clay and let go: most of the dent stays. The difference is not stiffness. It is that clay, once strained beyond a limit, changes the shape it wants to return to.

The standard way to write this is to split the strain $\varepsilon$ of the material into an elastic and a plastic part,

$$
\varepsilon = \varepsilon_e + \varepsilon_p ,
$$

and to let the stress depend on the elastic part only. The plastic part is a memory. It does not produce stress, and it changes only while the material is yielding.

### The elastic law

MuJoCo's 3D flex uses the Saint Venant–Kirchhoff model: the stored energy per unit volume is

$$
\Psi(\varepsilon_e) = \mu\,\operatorname{tr}(\varepsilon_e^2) + \frac{\lambda}{2}\,\operatorname{tr}(\varepsilon_e)^2,
\qquad
\mu = \frac{E}{2(1+\nu)},\quad \lambda = \frac{E\,\nu}{(1+\nu)(1-2\nu)}, \tag{1}
$$

with $\varepsilon$ the Green strain, so that a rigid rotation produces no stress. For this clay $E = 150$ kPa and $\nu = 0.3$, which gives $\mu = 57.7$ kPa and $\lambda = 86.5$ kPa.

### The yield bound and the return mapping

MuJoCo 4.0 adds the attribute `yield` to the `elasticity` element. Its documentation describes it as a von Mises ($J_2$) bound on the elastic strain, enforced by **radial return mapping**, with plastic flow that preserves volume. In words:

1. Take a step assuming all new strain is elastic (the trial state).
2. Split the elastic strain into a volumetric part and a **deviatoric** part $e = \varepsilon_e - \tfrac13\operatorname{tr}(\varepsilon_e)\,I$, the part that changes shape without changing volume.
3. If the size of $e$ exceeds the yield strain $\varepsilon_y$, scale $e$ back radially onto the bound and add what was removed to $\varepsilon_p$.

Only the deviatoric part can yield. A material squeezed equally from all sides does not flow; it flows when it is sheared.

The figure below runs that algorithm in one dimension: strain is driven up to a maximum and back to zero stress.

<figure class="widget" id="w-yield">
<div class="widget-controls">
<label for="w-yield-ey">yield strain</label><input id="w-yield-ey" type="range" min="0.5" max="6" step="0.1" value="2"><output id="w-yield-ey-out">2.0 %</output>
<label for="w-yield-em">largest strain applied</label><input id="w-yield-em" type="range" min="1" max="30" step="0.5" value="25"><output id="w-yield-em-out">25.0 %</output>
</div>
<svg id="w-yield-svg" viewBox="0 0 640 250" role="img" aria-label="Stress against strain for a loading and unloading cycle of an elastic, perfectly plastic material">
<line x1="60" x2="620" y1="210" y2="210" stroke="var(--sl-color-gray-4)" stroke-width="1.5"></line>
<line x1="60" x2="60" y1="20" y2="210" stroke="var(--sl-color-gray-4)" stroke-width="1.5"></line>
<text x="340" y="242" text-anchor="middle" font-size="12" fill="var(--sl-color-gray-3)">strain [%]</text>
<text x="20" y="115" text-anchor="middle" font-size="12" fill="var(--sl-color-gray-3)" transform="rotate(-90 20 115)">stress / (E · yield strain)</text>
<text x="52" y="214" text-anchor="end" font-size="11" fill="var(--sl-color-gray-3)">0</text>
<text x="52" y="64" text-anchor="end" font-size="11" fill="var(--sl-color-gray-3)">1</text>
<line x1="60" x2="620" y1="60" y2="60" stroke="var(--sl-color-hairline)" stroke-width="1" stroke-dasharray="4 4"></line>
<text x="60" y="226" text-anchor="middle" font-size="11" fill="var(--sl-color-gray-3)">0</text>
<text x="246.7" y="226" text-anchor="middle" font-size="11" fill="var(--sl-color-gray-3)">10</text>
<text x="433.3" y="226" text-anchor="middle" font-size="11" fill="var(--sl-color-gray-3)">20</text>
<text x="620" y="226" text-anchor="middle" font-size="11" fill="var(--sl-color-gray-3)">30</text>
<polyline id="w-yield-load" points="" fill="none" stroke="var(--sl-color-accent)" stroke-width="2.5" stroke-linejoin="round"></polyline>
<polyline id="w-yield-unload" points="" fill="none" stroke="var(--sl-color-text)" stroke-width="2.5" stroke-dasharray="6 4" stroke-linejoin="round"></polyline>
<line id="w-yield-set" x1="60" x2="60" y1="204" y2="216" stroke="var(--sl-color-text)" stroke-width="2.5"></line>
<line x1="440" x2="466" y1="34" y2="34" stroke="var(--sl-color-accent)" stroke-width="2.5"></line>
<text x="472" y="38" font-size="11.5" fill="var(--sl-color-text)">loading</text>
<line x1="440" x2="466" y1="51" y2="51" stroke="var(--sl-color-text)" stroke-width="2.5" stroke-dasharray="6 4"></line>
<text x="472" y="55" font-size="11.5" fill="var(--sl-color-text)">unloading</text>
</svg>
<div class="widget-summary">Elastic up to <strong id="w-yield-s1">–</strong> %, then the stress stays at the bound. After unloading, <strong id="w-yield-s2">–</strong> % of strain remains: <strong id="w-yield-s3">–</strong> % of what was applied.</div>
<figcaption>One-dimensional, ideal elasto-plasticity, schematic. The permanent strain is the largest strain minus the yield strain. With a yield strain of 2 % and a press that strains the top of the block by 25 %, the material law alone would keep 92 % of the stroke.</figcaption>
</figure>
<script>
(function(){
var eyI=document.getElementById('w-yield-ey'),emI=document.getElementById('w-yield-em');
function X(e){return 60+e*(560/30);}
function Y(s){return 210-s*150;}
function redraw(){
var ey=parseFloat(eyI.value),em=parseFloat(emI.value);
document.getElementById('w-yield-ey-out').textContent=ey.toFixed(1)+' %';
document.getElementById('w-yield-em-out').textContent=em.toFixed(1)+' %';
var peak=Math.min(em,ey)/ey,set=Math.max(em-ey,0);
var load=[X(0).toFixed(1)+','+Y(0).toFixed(1),X(Math.min(em,ey)).toFixed(1)+','+Y(peak).toFixed(1)];
if(em>ey){load.push(X(em).toFixed(1)+','+Y(1).toFixed(1));}
document.getElementById('w-yield-load').setAttribute('points',load.join(' '));
document.getElementById('w-yield-unload').setAttribute('points',X(em).toFixed(1)+','+Y(peak).toFixed(1)+' '+X(set).toFixed(1)+','+Y(0).toFixed(1));
var sl=document.getElementById('w-yield-set');
sl.setAttribute('x1',X(set).toFixed(1));sl.setAttribute('x2',X(set).toFixed(1));
document.getElementById('w-yield-s1').textContent=Math.min(em,ey).toFixed(1);
document.getElementById('w-yield-s2').textContent=set.toFixed(1);
document.getElementById('w-yield-s3').textContent=(100*set/em).toFixed(0);}
eyI.addEventListener('input',redraw);emI.addEventListener('input',redraw);redraw();
})();
</script>

:::caution[What is taken from the documentation and what was measured]
Equation (1) and the description of the yield bound are MuJoCo's documentation. The exact norm used for "the size of $e$" was not verified here, and nothing below depends on it. What this project relies on was measured directly: dents stay, the plastic strain lives in `mjData.plastic`, and restoring it reproduces a rollout bit for bit.
:::

## The block

| | |
|---|---|
| size | $140 \times 140 \times 40$ mm, bottom stuck to the board |
| mass | 1.18 kg (1,500 kg/m³) |
| Young's modulus, Poisson ratio | 150 kPa, 0.3 |
| yield strain | 0.02 |
| Rayleigh damping | 0.01 s |
| contact with the tool | frictionless, one contact dimension |
| integrator, timestep | `discrete` (implicit), 5 ms, conjugate-gradient solver |

The material and contact parameters started from the `clay.xml` and `roller.xml` examples in the MuJoCo repository. Two were changed for a reason.

**The clay must hold its own weight.** The stress at the bottom of the block from its weight is $\rho g h = 1500 \times 9.81 \times 0.04 = 590$ Pa. The deviatoric stress at yield is of order $2\mu\varepsilon_y = 2.3$ kPa, four times that. The margin matters: halving the yield strain raises the motion of the block after a press tenfold, from 0.015 mm to 0.12 mm (last table of this page), because gravity is then pushing a block that is close to its yield limit everywhere.

**The Poisson ratio is not zero.** The bulk modulus is $K = E / (3(1 - 2\nu))$. At $\nu = 0$ it is $E/3$, smaller than the shear modulus $E/2$: the material is easier to crush than to shear, and elements under a tool edge collapse. At $\nu = 0.3$, $K = 0.83\,E$ against $\mu = 0.38\,E$. The table below shows the difference on the full-vertex flex.

## Two ways to discretise, and why one was dropped

MuJoCo offers two kinds of 3D flex.

- **Full vertices.** Every vertex is a body with three degrees of freedom. The solid is cut into tetrahedra, each with its own plastic strain.
- **Trilinear cage** (`dof="trilinear"`). A coarse lattice of nodes carries the degrees of freedom, and a finer set of vertices, used for collision and drawing, is interpolated from it ([Nodes, Vertices, Height Map](/shape/representation/)). There is one plastic strain per cage cell.

The same press, with the same rounded tool on the same block, in both:

| discretisation | press | share of the stroke that stays | largest vertex motion in the 1.5 s after the primitive | smallest element volume / reference | inverted elements |
|---|---|---|---|---|---|
| **cage $12 \times 12 \times 3$ (used)** | 10 mm | 65 % | 0.015 mm | 0.86 | 0 |
| | 14 mm | 63 % | 0.005 mm | 0.83 | 0 |
| cage $12 \times 12 \times 2$ | 10 mm | 59 % | 0.025 mm | 0.89 | 0 |
| | 14 mm | 55 % | 0.063 mm | 0.86 | 0 |
| full vertices $15 \times 15 \times 4$, $\nu = 0.3$ | 10 mm | 81 % | 0.61 mm | 0.33 | 0 |
| | 14 mm | 100 % | 6.0 mm | −1.07 | 20 |
| full vertices $15 \times 15 \times 4$, $\nu = 0$ | 10 mm | 100 % | 3.1 mm | −1.12 | 21 |
| | 14 mm | 100 % | 6.5 mm | −1.50 | 61 |

<figure class="plot">
<img class="fig-light" src="/fig/m0-discretisation-light.svg" alt="Share of the stroke retained and creep after unloading for the cage and the full-vertex flex">
<img class="fig-dark" src="/fig/m0-discretisation-dark.svg" alt="Share of the stroke retained and creep after unloading for the cage and the full-vertex flex">
<figcaption>One press with the flat press at 10 and 14 mm. The right panel is logarithmic: the full-vertex block is still moving by millimetres a second and a half after the tool has left.</figcaption>
</figure>

The full-vertex flex keeps more of the stroke, which is what real clay does. But it does not come to rest. At 10 mm its vertices are still moving 0.6 mm in the second and a half after the tool has gone, and one element has been crushed to a third of its volume. At 14 mm twenty tetrahedra have **negative** volume: they have been pushed through themselves. The "100 %" in the table is not a good dent, it is a block collapsing under the place where the tool was. An inverted element has no meaningful stress, and the simulation that continues from there is not a simulation of anything.

The cage does none of this. The smallest cell keeps 83 % of its volume at the deepest press the planner is allowed, nothing inverts, and the block is at rest to a hundredth of a millimetre. It also has fewer degrees of freedom, 1,521 against 2,025.

A dynamics model, learned or not, needs a "next shape" that exists. That decided it.

## What the cage costs

Only two thirds of a stroke stays. The widget above says the material law would keep over 90 %. The difference is the discretisation, and the evidence is that it does not depend on the material:

| yield strain | share of a 10 mm stroke that stays (cage $12 \times 12 \times 3$) |
|---|---|
| 0.01 | 65 % (and 0.12 mm of motion after the press) |
| 0.02 (used) | 65 % (0.015 mm) |
| 0.04 | 63 % (0.001 mm) |

A fourfold change of the yield strain moves the result by two points. The reason is in how the cage stores plasticity. Each cell has **one** plastic strain tensor, a constant over the cell. A tool edge that crosses a cell bends and twists it, a deformation that varies across the cell and that no constant tensor can absorb. That part stays elastic and springs back when the tool leaves.

<figure class="plot">
<img class="fig-light" src="/fig/m0-permanence-light.svg" alt="Depth at the centre of a press against time for plastic and elastic clay">
<img class="fig-dark" src="/fig/m0-permanence-dark.svg" alt="Depth at the centre of a press against time for plastic and elastic clay">
<figcaption>One 10 mm press, then two seconds with the tool away. Plastic clay: 9.9 mm under the tool, 6.7 mm afterwards, and 0.012 mm of drift during the wait. The same block with the yield bound switched off returns to flat.</figcaption>
</figure>

So the spring-back in this project is a third of every stroke, it is a numerical property of a coarse cage and not a claim about clay, and it is exactly the kind of thing a geometric model of the tool gets wrong and a learned correction has to supply. Treat "the clay" in the rest of this site as *this simulated clay*.

Two more departures from an ideal plastic solid, both measured in milestone 0:

- **Volume is not conserved.** After three presses the cage has 97.6 % of its original volume, after three roller passes 94.9 %. Plastic flow preserves volume in the model, but the elastic part is compressible ($\nu = 0.3$), and a volume-preserving strain increment preserves volume only to first order, which at strains of 20 to 30 % leaves room for a few percent. Which of the two accounts for the loss was not separated.
- **The block bulges sideways** above its pinned base, by 7.5 mm after three presses and 12 mm after three roller passes. The height map of [part III](/shape/representation/) does not see the side walls, which matters in [part IV](/planning/oracle/).

:::tip[Where in the code]
`models/scene.py`: `ClayCfg` (every number in the table above), `_world_xml` (the `flexcomp`).
`scripts/m0_discretisation.py` → `results/m0_discretisation.json` (both tables); `scripts/m0_bench.py` → sections "permanence" and "sequences" of `results/m0_bench.json`.
Test: `test_dent_is_permanent` in `tests/test_bench.py`.
:::
