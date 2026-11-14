---
title: "The Geometric Prior"
description: "What is known without learning anything: the lower envelope of the volume a tool sweeps, written as one formula for three tools, and the shape update it implies for the clay."
sidebar:
  order: 1
---

:::note[Guiding question]
Before any data is collected, what can be said about the shape of the clay after a press? And which parts of the answer are certain, and which are guesses about the material?
:::

## What is certain

Two things are known exactly and have nothing to do with clay: the shape of the tool head, and where the arm takes it ([Action Primitives](/bench/primitives/)). Together they define the **swept volume**, every point in space the head occupies at some moment of the primitive. Whatever the material does, it is not inside that volume while the tool is at the bottom of its stroke.

What is *not* known is everything else: where the displaced material goes, how much of the dent springs back when the tool leaves, and what happens around the edge of the contact. The prior makes the simplest possible guess for those and leaves the corrections to a network.

## The lower envelope of a tool

Only the underside of the swept volume matters for a tool that comes from above. Define the **envelope** $z_{\text{env}}(X, Y)$ as the lowest height the head reaches above the point $(X, Y)$ of the table at any time during the primitive.

All three heads are built from the same kind of solid: a flat **footprint** in a horizontal plane, thickened downwards by a ball of radius $r$. For such a solid at the bottom of its stroke, with its underside at height $z_b = z_{\text{top}} - d$,

$$
z_{\text{env}}(X, Y) =
\begin{cases}
z_b + r - \sqrt{r^2 - \delta^2}, & \delta < r,\\[2pt]
+\infty, & \delta \ge r,
\end{cases}
\tag{1}
$$

where $\delta$ is the horizontal distance from $(X, Y)$ to the footprint. Directly under the footprint $\delta = 0$ and the envelope is flat at $z_b$. Within $r$ of it the envelope rises along a circular arc. Further out the tool is never above the point at all.

The distance $\delta$ is computed in the tool frame. For an action $a = (x, y, \psi, d, \ell)$, a table point has tool-frame coordinates

$$
l_x = \cos\psi\,(X - x) + \sin\psi\,(Y - y),\qquad l_y = \sin\psi\,(X - x) - \cos\psi\,(Y - y),
$$

and for a rectangular footprint with half-sizes $(f_x, f_y)$,

$$
\delta = \sqrt{\max(|l_x| - f_x,\, 0)^2 + \max(|l_y| - f_y,\, 0)^2}.
$$

The three tools differ only in the footprint:

| tool | footprint half-sizes $(f_x, f_y)$ | radius $r$ |
|---|---|---|
| flat press | $(12, 12)$ mm: the $40 \times 40$ mm face minus its 8 mm rounding | 8 mm |
| cross stamp | two rectangles, $(34, 3)$ mm and $(3, 34)$ mm; the envelope is the lower of the two | 6 mm |
| roller | the axis of the capsule, **swept along the travel**: $(25\ \text{mm},\ \ell/2)$ | 12 mm |

The roller's line is the whole point of writing the envelope this way. A capsule is a segment thickened by a ball. Sliding it sideways by $\ell$ sweeps the segment into a rectangle, and the swept volume is that rectangle thickened by the same ball. So the envelope of the whole rolling motion is formula (1) with a larger footprint, and no integration over time is needed.

### The clay is made of spheres

A flex vertex collides as a sphere of radius $\rho = 2$ mm. Its *centre* must stay below the envelope of the tool inflated by $\rho$, which is the same kind of solid with a larger radius:

$$
z^{\rho}_{\text{env}}(X, Y) = (z_b - \rho) + (r + \rho) - \sqrt{(r + \rho)^2 - \delta^2},\qquad \delta < r + \rho. \tag{2}
$$

### Checked against the simulated tool

Formula (1) is a claim about geometry, so it is tested against geometry. The arm is placed at the bottom pose of a primitive, rays are cast upwards from 400 random table points, and the height at which each ray meets the tool head is compared with (1). For all three tools and three yaws the footprints agree on at least 97 % of the points (the rest lie within a ray's width of the outline), and the heights agree to 0.3 mm under the flat part and 0.2 mm on average over the rounded rim. For the roller the same comparison is made against the lowest hit over 71 positions along its travel. The residual is the facetting of the convex meshes.

This test also pins down every sign convention at once: the yaw, the direction of travel, and which way the tool frame's axes point.

## From the envelope to a shape

The prior turns the envelope into a prediction of the next shape in three steps.

**1. Clip the top surface.** Every vertex of the top layer is pushed down to the inflated envelope if it is above it, and left alone otherwise:

$$
z'_{\text{top}} = \min\big(z_{\text{top}},\; z^{\rho}_{\text{env}}(x, y)\big). \tag{3}
$$

**2. Compress the column.** The bottom of the block is pinned. The vertices below a clipped one are moved in proportion, as if the column were squeezed uniformly:

$$
z'_k = z_0 + (z_k - z_0)\,\frac{z'_{\text{top}} - z_0}{z_{\text{top}} - z_0}, \tag{4}
$$

with $z_0$ the height of the pinned vertex of that column.

**3. Carry the displacement to the nodes.** The state of the clay is the cage nodes, not the vertices ([Nodes, Vertices, Height Map](/shape/representation/)). With $W$ the interpolation matrix and $\Delta v$ the vertex displacement of steps 1 and 2, each node takes the weighted mean of the displacements of the vertices it influences:

$$
\Delta n_i = \frac{\sum_v W_{vi}\,\Delta v_v}{\sum_v W_{vi}}. \tag{5}
$$

Nothing moves sideways in any of the three steps.

Step 3 loses something, and the loss is real, not an artefact of the prior. A cage with nodes 11.7 mm apart cannot hold a dent with a sharp rim, whatever the tool does. The figure below is a one-dimensional section through a press: drag the sliders to see what the clipping asks for and what the cage can hold.

<figure class="widget" id="w-env">
<div class="widget-controls">
<label for="w-env-d">depth d</label><input id="w-env-d" type="range" min="2" max="14" step="0.5" value="10"><output id="w-env-d-out">10.0 mm</output>
<label for="w-env-a">tool half-width</label><input id="w-env-a" type="range" min="9" max="30" step="1" value="9"><output id="w-env-a-out">9 mm</output>
<label for="w-env-r">edge radius r</label><input id="w-env-r" type="range" min="1" max="12" step="0.5" value="6"><output id="w-env-r-out">6.0 mm</output>
<label for="w-env-x">tool position</label><input id="w-env-x" type="range" min="-12" max="12" step="0.5" value="5.5"><output id="w-env-x-out">5.5 mm</output>
</div>
<svg id="w-env-svg" viewBox="0 0 640 250" role="img" aria-label="Section through a press: the tool, the clipped vertices and the surface the cage can represent">
<line x1="30" x2="610" y1="110" y2="110" stroke="var(--sl-color-gray-4)" stroke-width="1" stroke-dasharray="4 4"></line>
<text x="34" y="104" font-size="11" fill="var(--sl-color-gray-3)">untouched surface</text>
<polygon id="w-env-tool" points="" fill="var(--sl-color-gray-5)" stroke="var(--sl-color-gray-3)" stroke-width="1.5"></polygon>
<polyline id="w-env-cage" points="" fill="none" stroke="var(--sl-color-accent)" stroke-width="2.5" stroke-linejoin="round"></polyline>
<g id="w-env-verts"></g>
<g id="w-env-nodes"></g>
<line x1="30" x2="610" y1="236" y2="236" stroke="var(--sl-color-gray-4)" stroke-width="1.5"></line>
<text x="320" y="248" text-anchor="middle" font-size="11" fill="var(--sl-color-gray-3)">section through the block, 140 mm wide; heights drawn 1.5 times larger than widths</text>
<circle cx="440" cy="20" r="3.2" fill="var(--sl-color-text)"></circle>
<text x="450" y="24" font-size="11" fill="var(--sl-color-text)">clipped vertices, equation (3)</text>
<rect x="436" y="33" width="8" height="8" fill="var(--sl-color-accent)"></rect>
<text x="450" y="41" font-size="11" fill="var(--sl-color-text)">cage nodes and surface, equation (5)</text>
</svg>
<div class="widget-summary">The tool bottom is <strong id="w-env-s1">–</strong> mm below the untouched surface. The deepest clipped vertex is at <strong id="w-env-s2">–</strong> mm; the deepest point the cage holds is <strong id="w-env-s3">–</strong> mm, <strong id="w-env-s4">–</strong> % of the stroke.</div>
<figcaption>One-dimensional section, schematic: 25 vertices 5.8 mm apart, a cage node at every second one. It opens on one bar of the stamp (9 mm half-width, 6 mm radius) placed halfway between two nodes: the cage holds about 70 % of the stroke. Centred on a node the same bar keeps nearly all of it, and the flat press (20 mm, 8 mm) keeps all of it anywhere.</figcaption>
</figure>
<script>
(function(){
var NS='http://www.w3.org/2000/svg',TOP=44,RHO=2,S=140/24;
var ids=['d','a','r','x'],inp={};
ids.forEach(function(k){inp[k]=document.getElementById('w-env-'+k);});
function X(mm){return 30+(mm+70)*(580/140);}
function Y(h){return 236-(h-23)*6.0;}
function env(x,x0,a,r,d,infl){var rr=r+infl,dist=Math.max(Math.abs(x-x0)-(a-r),0);
if(dist>=rr){return Infinity;}
return TOP-d-infl+rr-Math.sqrt(rr*rr-dist*dist);}
function redraw(){
var d=parseFloat(inp.d.value),a=parseFloat(inp.a.value),r=Math.min(parseFloat(inp.r.value),a),x0=parseFloat(inp.x.value);
document.getElementById('w-env-d-out').textContent=d.toFixed(1)+' mm';
document.getElementById('w-env-a-out').textContent=a.toFixed(0)+' mm';
document.getElementById('w-env-r-out').textContent=r.toFixed(1)+' mm';
document.getElementById('w-env-x-out').textContent=x0.toFixed(1)+' mm';
var pts=[],i,x;
pts.push(X(x0-a).toFixed(1)+','+Y(60).toFixed(1));
for(i=0;i<=60;i++){x=x0-a+2*a*i/60;var z=env(x,x0,a,r,d,0);
if(!isFinite(z)){z=TOP-d+r;}
pts.push(X(x).toFixed(1)+','+Y(z).toFixed(1));}
pts.push(X(x0+a).toFixed(1)+','+Y(60).toFixed(1));
document.getElementById('w-env-tool').setAttribute('points',pts.join(' '));
var hv=[],dv=[];
for(i=0;i<25;i++){x=-70+i*S;var c=TOP-RHO,e=env(x,x0,a,r,d,RHO),zc=Math.min(c,e);
hv.push(zc+RHO);dv.push(zc-c);}
var gv=document.getElementById('w-env-verts');
while(gv.firstChild){gv.removeChild(gv.firstChild);}
for(i=0;i<25;i++){var ci=document.createElementNS(NS,'circle');
ci.setAttribute('cx',X(-70+i*S).toFixed(1));ci.setAttribute('cy',Y(hv[i]).toFixed(1));ci.setAttribute('r','3.2');
ci.setAttribute('fill','var(--sl-color-text)');gv.appendChild(ci);}
var gn=document.getElementById('w-env-nodes'),cage=[],deep=0;
while(gn.firstChild){gn.removeChild(gn.firstChild);}
for(i=0;i<13;i++){var v=2*i,num=dv[v],den=1;
if(v>0){num+=0.5*dv[v-1];den+=0.5;}
if(v<24){num+=0.5*dv[v+1];den+=0.5;}
var hn=TOP+num/den;deep=Math.max(deep,TOP-hn);
cage.push(X(-70+v*S).toFixed(1)+','+Y(hn).toFixed(1));
var re=document.createElementNS(NS,'rect');
re.setAttribute('x',(X(-70+v*S)-4).toFixed(1));re.setAttribute('y',(Y(hn)-4).toFixed(1));
re.setAttribute('width','8');re.setAttribute('height','8');re.setAttribute('fill','var(--sl-color-accent)');gn.appendChild(re);}
document.getElementById('w-env-cage').setAttribute('points',cage.join(' '));
var deepV=0;
for(i=0;i<25;i++){deepV=Math.max(deepV,TOP-hv[i]);}
document.getElementById('w-env-s1').textContent=d.toFixed(1);
document.getElementById('w-env-s2').textContent=deepV.toFixed(1);
document.getElementById('w-env-s3').textContent=deep.toFixed(1);
document.getElementById('w-env-s4').textContent=(100*deep/d).toFixed(0);}
ids.forEach(function(k){inp[k].addEventListener('input',redraw);});
redraw();
})();
</script>

:::tip[Where in the code]
`learn/prior.py`: `tool_frame`, `envelope` (equations (1) and (2) for all three tools), `SweptVolumePrior` (steps (3) to (5); `projection="lsq"` gives the least-squares alternative for comparison). `models/scene.py`: `TOOL_GEOMS`, the head dimensions shared by the simulated tools and the prior.
Tests: `tests/test_prior.py` (the envelope against ray casts on the simulated tools), `test_prior_never_raises_and_only_acts_under_the_tool` and `test_prior_beats_no_change_on_a_real_press` in `tests/test_models.py`.
:::
