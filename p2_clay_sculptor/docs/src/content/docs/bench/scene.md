---
title: "Scene and Tools"
description: "A Menagerie Panda with three interchangeable tool heads assembled with MjSpec, the contact filtering that keeps the clay away from everything but the tool, and a servo that actually reaches the commanded depth."
sidebar:
  order: 2
---

:::note[Guiding question]
The brief says the tool and arm kinematics are "known exactly". An arm pushing 30 N into clay through position servos does not go where it is told. How much of the commanded depth does it reach, and what does it take to make "known exactly" true enough to build a model on?
:::

## What is in the world

- A **Franka Panda** from MuJoCo Menagerie, the hand-less model, with its base at the origin.
- A **clay block**, $140 \times 140 \times 40$ mm, centred 0.5 m in front of the base and stuck to a board ([The Clay](/bench/clay/)).
- One of three **tool heads** bolted to the flange, 100 mm long from the flange to the bottom of the head.

| tool | head | what it is for |
|---|---|---|
| flat press | $40 \times 40$ mm face, bottom edges rounded with an 8 mm radius | pockets and flats |
| roller | a capsule of 12 mm radius, 74 mm long overall, axis horizontal | bands and slabs |
| cross stamp | two bars $80 \times 18$ mm at right angles, 6 mm edge radius | a cross, or an X when turned 45° |

The roller does not spin. Contact between tool and clay is frictionless, and a frictionless cylinder that slides leaves the same shape as one that rolls.

### Why the edges are rounded

A sharp-edged box pressed into a coarse finite-element block concentrates all the shear into the one row of elements under its edge, and elements that are sheared and crushed far enough turn inside out ([The Clay](/bench/clay/) has the measurement). Rounding the edges with a radius comparable to the element size spreads that shear over several elements. The rounded heads are convex meshes generated from a formula, and the same formula is the [geometric prior](/dynamics/prior/): the simulated tool and the model of the tool cannot drift apart.

## Assembling the scene with MjSpec

MuJoCo 4.0's `MjSpec` builds models programmatically and attaches one model to another. The scene is put together in three steps.

```python
world = mujoco.MjSpec.from_string(world_xml)          # floor, board, lights, cameras, the clay flexcomp
panda = mujoco.MjSpec.from_file("panda_nohand.xml")   # Menagerie
world.attach(panda, frame=world.frame("robot_base"), prefix="panda_")
world.attach(tool_spec(name), site=world.site("panda_attachment_site"), prefix="tool_")
model = world.compile()
```

The clay is written as XML because `flexcomp` is a macro of the XML parser. The tool is its own small spec: a body with a visual stem, one or two collision geoms for the head, and a site `tip` at the centre of the bottom face.

Each tool gives a separately compiled model. The three models differ only in geoms welded to the flange, so they have the same joints and the same state layout, and a tool change is a change of model with the state carried over unchanged ([State, Save and Restore](/bench/state/)).

Three changes are made to Menagerie's Panda, and nothing else:

1. **Gravity compensation** is switched on for every arm and tool body (`gravcomp = 1`), so the servos carry no gravity load.
2. The seven **position servos are replaced** by `pid` actuators (below).
3. The keyframe is deleted, because the number of coordinates changes when the clay is added.

### Contact filtering

The clay and the tool heads are given contact bit 4, and nothing else has it. The clay can therefore touch a tool head and nothing else: not the arm, not the floor (its bottom layer is pinned, so it needs no floor). The tool head can touch nothing but the clay. This keeps the pinned bottom layer from holding a permanent contact with the floor at every vertex, and makes "the only thing that deforms the clay is the tool head" a property of the model.

## A servo that reaches the commanded depth

Menagerie's Panda has position servos of the form

$$
\tau = k_p\,(q^{\text{ref}} - q) - k_v\,\dot q, \tag{1}
$$

with $k_p = 4500$ N m/rad and $k_v = 450$ N m s/rad on the shoulder joints. Two things are wrong with (1) for this task.

**It lags a moving reference.** The damping acts on the joint velocity, not on the velocity *error*. To follow a reference moving at constant speed $\omega$ the servo must supply $\tau = 0$ in free space, which (1) does only when $k_p\,e = k_v\,\omega$: a steady lag of

$$
e = \frac{k_v}{k_p}\,\omega = 0.1\ \text{s} \times \omega .
$$

At the 136 mm/s peak of a descent that is more than a centimetre.

**It gives way under load.** Holding the tool against a contact torque $\tau_c$ needs $k_p\,e = \tau_c$, a steady error of $\tau_c / k_p$. A press loads joint 2 with 24 N m, which is 5 mrad, or about 2 mm at the tool.

MuJoCo 4.0 has a `pid` actuator with position and velocity setpoints and integral action:

$$
\tau = k_p\,(q^{\text{ref}} - q) + k_v\,(\dot q^{\text{ref}} - \dot q) + k_i \int (q^{\text{ref}} - q)\,dt. \tag{2}
$$

The velocity setpoint removes the lag, and the integral removes the steady error under load. Menagerie's $k_p$ and $k_v$ are kept. The integral gain is $k_i = k_p / T_i$ with $T_i = 0.05$ s, and the integrator is clamped at 0.02 rad s against wind-up. Each joint now takes two controls, so the arm has 14 controls and 7 integrator states.

The same press, commanded to 10 mm, with and without the integral term:

| servo | depth reached | tip error at the end of the dwell (x, z) | dent left in the clay |
|---|---|---|---|
| PD on position and velocity error, no integral | 8.4 mm | 1.75 mm, 1.70 mm | 5.5 mm |
| `pid`, equation (2) | 10.0 mm | 0.17 mm, 0.00 mm | 6.7 mm |

Without the integral the arm is 1.7 mm short and the dent is a sixth shallower. That shortfall would not be a property of clay. A dynamics model trained on such data would learn the arm's compliance mixed into the clay's response, and the geometric prior, which assumes the tool goes where it is sent, would be wrong by the same amount.

### What is left

Tracking is not perfect, and the numbers are part of what the learned models have to absorb.

| tool | tip error at the end of the dwell (x, y, z) [mm] | largest error during the primitive [mm] | largest joint torque |
|---|---|---|---|
| press | −0.17, +0.01, +0.00 | 1.6 | 24 N m |
| stamp | −0.74, +0.05, −0.52 | 2.9 | 33 N m |
| roller | −0.29, −0.03, −0.09 | 1.6 | 24 N m |

The largest errors are transients on first contact, split about equally between $x$ and $z$: the tool is forward of and below the shoulder, so when the load rotates the arm about the shoulder the tool is lifted and pushed away from the base in equal measure. The stamp, with the largest contact area and so the largest force, ends its dwell 0.5 mm *below* the reference and 0.7 mm towards the base: the integrator wound up while the head was sinking in, and 0.1 s is not enough for it to unwind. Its deepest point is 10.6 mm for a commanded 10.0 mm; the roller reaches 10.3 mm during its slide. So "the kinematics are known exactly" holds to a few tenths of a millimetre for the press and the roller and to about 0.7 mm for the stamp.

:::tip[Where in the code]
`models/scene.py`: `build_spec` (the three attachment steps and the Panda changes), `_tool_spec`, `TOOL_GEOMS` (the head dimensions, shared with the prior), `rounded_box_vertices`, `Index`.
Numbers: `scripts/m0_bench.py`, sections "tracking" and "tracking_pd_only" of `results/m0_bench.json`.
:::
