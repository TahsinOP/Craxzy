"""Scene construction (MuJoCo 4.0): a plastic clay block, a Franka Panda and three tools.

The scene is assembled with ``MjSpec``:

* the world (floor, board, lights, cameras and the clay ``flexcomp``) is parsed from
  an XML string, because ``flexcomp`` is an XML-level macro;
* Menagerie's hand-less Panda is attached to a frame at the robot base;
* one tool (a separate ``MjSpec``) is attached to the Panda's ``attachment_site``.

One compiled model per tool.  The three models differ only in the tool geoms that
are welded to the flange, so ``nq``, ``nv``, ``nu`` and ``nplastic`` are identical
and a full physics state of one model is a valid state of the others: changing the
tool is copying a state vector.

Clay: a 3D flex with a trilinear cage (``dof="trilinear"``) whose bottom node layer
is pinned to the board, with ``<elasticity yield=...>``.  The plastic strain (one
symmetric tensor per cage cell) lives in ``mjData.plastic``.

Changes to Menagerie's Panda (nothing else is touched):

* ``gravcomp = 1`` on every arm and tool body, so the servos carry no gravity load;
* the seven position servos (PD on position with damping on the joint velocity,
  which lags a moving reference by kv/kp = 0.1 s) are replaced by MuJoCo 4.0 ``pid``
  actuators with position and velocity setpoints and integral action: the force is
  kp (q_ref - q) + kv (qd_ref - qd) + ki * integral(q_ref - q).  Menagerie's kp and
  kv are kept (times ``gain_scale``); torque limits are kept;
* the keyframe is deleted (``nq`` changes when the clay is added).

Control vector: for each joint ``[q_ref, qd_ref]``, so ``nu = 14``.

Collision filtering: the clay and the tool heads use contact bit 4; nothing else
does, so the clay touches the tool head and nothing else, and the tool touches
nothing but the clay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

PREFIX = "panda_"
ARM_JOINTS = [f"{PREFIX}joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"{PREFIX}actuator{i}" for i in range(1, 8)]
HOME_Q = np.array([0.0, 0.25, 0.0, -2.35, 0.0, 2.60, 0.785])   # tool pointing down above the clay
CLAY_BIT = 4
HEAD_GROUP = 1            # geom group of the tool-head collision geoms (ray casts in the tests)
TOOL_LENGTH = 0.10        # flange -> lowest point of every tool head [m]
TOOLS = ("press", "roller", "stamp")


@dataclass(frozen=True)
class ClayCfg:
    """Geometry, discretisation and material of the clay block (SI units)."""

    centre: tuple = (0.50, 0.0)          # block centre on the table plane (robot base at the origin)
    size: tuple = (0.14, 0.14, 0.04)     # block extents x, y, z
    cells: tuple = (12, 12, 3)           # trilinear cage cells (the dynamic degrees of freedom)
    count: tuple = (25, 25, 4)           # interpolated vertices (collision and shape)
    radius: float = 0.002                # vertex collision radius
    density: float = 1500.0              # kg/m^3
    young: float = 1.5e5                 # Pa
    poisson: float = 0.3
    yield_strain: float = 0.02           # von Mises bound on the elastic deviatoric strain
    damping: float = 0.01                # Rayleigh damping coefficient [s]
    solref: tuple = (0.004, 1.0)
    solimp: tuple = (0.98, 0.995, 0.001)
    dof: str = "trilinear"               # "trilinear" (cage) or "full" (every vertex a body; only for scripts/m0_discretisation.py)

    @property
    def mass(self) -> float:
        return self.density * self.size[0] * self.size[1] * self.size[2]

    @property
    def spacing(self) -> tuple:
        return tuple(self.size[i] / (self.count[i] - 1) for i in range(3))

    @property
    def top(self) -> float:
        """Height of the undeformed top surface (vertex centres + collision radius)."""
        return self.size[2] + 2 * self.radius

    @property
    def z_centre(self) -> float:
        return self.size[2] / 2 + self.radius


@dataclass(frozen=True)
class SimCfg:
    timestep: float = 0.005
    solver: str = "CG"
    iterations: int = 150
    tolerance: float = 1e-5
    gain_scale: float = 1.0              # multiplies the Menagerie servo gains (kp and kv)
    ki_time: float = 0.05                # integral time [s]: ki = kp / ki_time (0 disables integral action)
    imax: float = 0.02                   # anti-windup bound on the integrated position error [rad s]


@dataclass(frozen=True)
class ToolGeom:
    """Analytic description of a tool head, shared by the MJCF builder and the
    geometric prior (``clay_sculptor.learn.prior``).  A head is a union of
    "stadium" solids: a segment of half-length ``half_len`` along the local x axis
    (rotated by ``yaw`` about the tool axis), widened to ``half_width`` and rounded
    with radius ``r`` at the bottom edges."""

    name: str
    parts: tuple = field(default_factory=tuple)   # tuples (kind, half_len, half_width, r, yaw)


TOOL_GEOMS = {
    # flat press: 40 x 40 mm face, 8 mm edge radius
    "press": ToolGeom("press", (("rbox", 0.020, 0.020, 0.008, 0.0),)),
    # roller: capsule of radius 12 mm, 50 mm between the cap centres, axis along tool x
    "roller": ToolGeom("roller", (("capsule", 0.025, 0.012, 0.012, 0.0),)),
    # cross stamp: two bars 80 x 18 mm, 6 mm edge radius
    "stamp": ToolGeom("stamp", (("rbox", 0.040, 0.009, 0.006, 0.0), ("rbox", 0.040, 0.009, 0.006, np.pi / 2))),
}


def rounded_box_vertices(a: float, b: float, r: float, h: float, n: int = 7) -> np.ndarray:
    """Vertices whose convex hull is the box [-a,a] x [-b,b] x [0,h] with the four
    bottom edges and corners rounded with radius ``r`` (bottom face at z = 0)."""
    pts = []
    th = np.linspace(0.0, np.pi / 2, n)     # polar angle measured from -z
    ph = np.linspace(0.0, np.pi / 2, n)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for t in th:
                for p in ph:
                    pts.append([sx * ((a - r) + r * np.sin(t) * np.cos(p)),
                                sy * ((b - r) + r * np.sin(t) * np.sin(p)),
                                r - r * np.cos(t)])
            pts.append([sx * a, sy * b, h])
    return np.asarray(pts)


@lru_cache(maxsize=1)
def panda_dir() -> Path:
    import mujoco_menagerie as mm

    return Path(mm.prefetch(["franka_emika_panda"], progress=None)[0])


def _world_xml(clay: ClayCfg | None, sim: SimCfg) -> str:
    flex = ""
    if clay is not None:
        c, s = clay, clay.spacing
        if c.dof == "trilinear":
            dof = f'dof="trilinear" cellcount="{c.cells[0]} {c.cells[1]} {c.cells[2]}"'
            pin = f"0 0 0 {c.cells[0]} {c.cells[1]} 0"
        else:
            dof, pin = "", f"0 0 0 {c.count[0] - 1} {c.count[1] - 1} 0"
        flex = f"""
    <flexcomp name="clay" type="grid" dim="3" {dof} count="{c.count[0]} {c.count[1]} {c.count[2]}"
              spacing="{s[0]:.9g} {s[1]:.9g} {s[2]:.9g}"
              pos="{c.centre[0]} {c.centre[1]} {c.z_centre}" radius="{c.radius}" mass="{c.mass:.9g}"
              material="clay" flatskin="false">
      <pin gridrange="{pin}"/>
      <contact condim="1" solref="{c.solref[0]} {c.solref[1]}" solimp="{c.solimp[0]} {c.solimp[1]} {c.solimp[2]}"
               selfcollide="none" contype="{CLAY_BIT}" conaffinity="{CLAY_BIT}"/>
      <elasticity young="{c.young:.9g}" poisson="{c.poisson}" damping="{c.damping}" yield="{c.yield_strain}"/>
    </flexcomp>"""
    cx = clay.centre[0] if clay is not None else 0.5
    bh = clay.size[0] / 2 + 0.035 if clay is not None else 0.105      # board half-width: 0.105 for the default block
    return f"""<mujoco model="clay_sculptor">
  <compiler autolimits="true" angle="radian"/>
  <option timestep="{sim.timestep}" solver="{sim.solver}" iterations="{sim.iterations}"
          tolerance="{sim.tolerance}" integrator="discrete"/>
  <size memory="32M"/>
  <visual>
    <global offwidth="1280" offheight="720" elevation="-28" azimuth="140"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <map znear="0.01"/>
  </visual>
  <statistic center="{cx} 0 0.1" extent="0.7" meansize="0.03"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.86 0.88 0.90" rgb2="0.62 0.66 0.72" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.81 0.82" rgb2="0.74 0.75 0.77" width="200" height="200"/>
    <material name="floor" texture="grid" texrepeat="6 6" texuniform="true" reflectance="0"/>
    <material name="board" rgba="0.24 0.26 0.30 1" specular="0.3" shininess="0.3"/>
    <material name="clay" rgba="0.78 0.40 0.24 1" specular="0.05" shininess="0.05"/>
    <material name="steel" rgba="0.62 0.65 0.70 1" specular="0.7" shininess="0.6"/>
    <material name="brass" rgba="0.82 0.66 0.34 1" specular="0.8" shininess="0.7"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="-0.5 -0.8 1.2" dir="0.5 0.6 -0.7" diffuse="0.7 0.68 0.64" specular="0.2 0.2 0.2"/>
    <light name="fill" directional="true" pos="1.2 0.6 0.9" dir="-0.7 -0.4 -0.5" diffuse="0.25 0.27 0.32" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    <geom name="board" type="box" pos="{cx} 0 -0.003" size="{bh} {bh} 0.004" material="board" contype="0" conaffinity="0"/>
    <camera name="wide" pos="{cx + 0.62} -0.62 0.52" xyaxes="0.72 0.69 0 -0.33 0.34 0.88" fovy="42"/>
    <camera name="close" pos="{cx + 0.22} -0.25 0.24" xyaxes="0.75 0.66 0 -0.36 0.41 0.84" fovy="40"/>
    <camera name="top" pos="{cx} 0 0.45" xyaxes="0 -1 0 1 0 0" fovy="28"/>{flex}
    <frame name="robot_base" pos="0 0 0"/>
  </worldbody>
</mujoco>"""


def _tool_spec(name: str) -> mujoco.MjSpec:
    """A tool as its own spec.  Tool frame: z along the flange axis, head bottom at
    z = TOOL_LENGTH, site ``tip`` at the centre of the bottom face."""
    g = TOOL_GEOMS[name]
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="tool")
    body.gravcomp = 1.0

    def visual(**kw):
        geom = body.add_geom(contype=0, conaffinity=0, density=0, **kw)
        return geom

    visual(name="flange", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, 0.008, 0], pos=[0, 0, 0.008],
           rgba=[0.25, 0.25, 0.27, 1])
    head_h = 0.030
    visual(name="stem", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.011, (TOOL_LENGTH - head_h - 0.016) / 2 + 0.004, 0],
           pos=[0, 0, 0.016 + (TOOL_LENGTH - head_h - 0.016) / 2], rgba=[0.62, 0.65, 0.70, 1])
    contact = dict(contype=CLAY_BIT, conaffinity=CLAY_BIT, condim=1, priority=1, solref=[0.003, 1.0],
                   solimp=[0.98, 0.995, 0.001, 0.5, 2], rgba=[0.82, 0.66, 0.34, 1], group=HEAD_GROUP)
    for k, (kind, hl, hw, r, yaw) in enumerate(g.parts):
        quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        if kind == "rbox":
            mesh = spec.add_mesh(name=f"head{k}")
            v = rounded_box_vertices(hl, hw, r, head_h)
            v[:, 2] = -v[:, 2]                      # rounded face towards +z (away from the flange)
            mesh.uservert = v.astype(np.float32).ravel()
            body.add_geom(name=f"head{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"head{k}",
                          pos=[0, 0, TOOL_LENGTH], quat=quat, mass=0.15, **contact)
        elif kind == "capsule":
            # axis along local x: rotate the capsule's z axis onto x, then yaw about the tool axis
            qy = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.asarray(quat), qy)
            body.add_geom(name=f"head{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, hl, 0],
                          pos=[0, 0, TOOL_LENGTH - r], quat=q, mass=0.15, **contact)
        else:
            raise ValueError(kind)
    body.add_geom(name="inertia", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, 0.04], mass=0.2,
                  contype=0, conaffinity=0, rgba=[0, 0, 0, 0])
    body.add_site(name="tip", pos=[0, 0, TOOL_LENGTH], size=[0.003, 0.003, 0.003], rgba=[1, 0, 0, 0.5], group=4)
    return spec


RACK_Y, RACK_TOP = 0.28, 0.06                                  # tool rack beside the clay [m]
RACK_SLOTS = {"roller": 0.38, "press": 0.50, "stamp": 0.62}      # x of each tool's pedestal


def add_rack(world: mujoco.MjSpec) -> None:
    """A rack with a copy of each tool standing on a pedestal, tip on the pedestal top and
    yaw 0 (the orientation the tool has on the flange at yaw 0).  Visual only: no geom
    collides and no body moves, so the state layout and the physics are unchanged.  Used
    by ``scripts/demo_toolchange.py`` to show tool changes."""
    wb = world.worldbody
    vis = dict(contype=0, conaffinity=0)
    wb.add_geom(name="rack_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.20, 0.06, 0.006],
                pos=[0.50, RACK_Y, 0.006], rgba=[0.30, 0.32, 0.36, 1], **vis)
    for t, x in RACK_SLOTS.items():
        wb.add_geom(name=f"rack_post_{t}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.024, RACK_TOP / 2, 0],
                    pos=[x, RACK_Y, RACK_TOP / 2], rgba=[0.45, 0.47, 0.52, 1], **vis)
        frame = wb.add_frame(name=f"rack_{t}_frame", pos=[x, RACK_Y, RACK_TOP + TOOL_LENGTH], quat=[0, 1, 0, 0])
        world.attach(_tool_spec(t), frame=frame, prefix=f"rack_{t}_")
    for g in world.geoms:
        if g.name.startswith("rack_"):
            g.contype, g.conaffinity = 0, 0


def build_spec(tool: str = "press", clay: ClayCfg | None = ClayCfg(), sim: SimCfg = SimCfg(),
               rack: bool = False) -> mujoco.MjSpec:
    """World + Panda + ``tool``.  ``clay=None`` gives the arm-only model used for
    kinematics (same arm, same tool, no flex).  ``rack`` adds the visual tool rack."""
    world = mujoco.MjSpec.from_string(_world_xml(clay, sim))

    panda = mujoco.MjSpec.from_file(str(panda_dir() / "panda_nohand.xml"))
    panda.meshdir = str(panda_dir() / "assets")
    for k in list(panda.keys):
        panda.delete(k)
    for b in panda.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    panda.option.integrator = world.option.integrator      # silence the attach conflict report:
    panda.option.impratio = world.option.impratio          # the world's options are the ones kept
    for i in range(1, 8):
        a = panda.actuator(f"actuator{i}")
        kp = float(a.gainprm[0]) * sim.gain_scale
        kv = -float(a.biasprm[2]) * sim.gain_scale
        frange = np.array(a.forcerange)
        a.set_to_pid(kp=kp, kv=kv, ki=kp / sim.ki_time if sim.ki_time > 0 else 0.0, imax=sim.imax)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = frange
    world.attach(panda, frame=world.frame("robot_base"), prefix=PREFIX)
    world.attach(_tool_spec(tool), site=world.site(PREFIX + "attachment_site"), prefix="tool_")
    if rack:
        add_rack(world)
    world.modelname = f"clay_sculptor_{tool}"
    return world


def build_model(tool: str = "press", clay: ClayCfg | None = ClayCfg(), sim: SimCfg = SimCfg()) -> mujoco.MjModel:
    return build_spec(tool, clay, sim).compile()


@dataclass
class Index:
    """Addresses that the rest of the code needs, looked up once per model."""

    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    arm_ctrl: np.ndarray        # (7, 2): addresses of [q_ref, qd_ref] per joint
    tip_site: int
    vert_adr: int
    vert_num: int

    @classmethod
    def of(cls, m: mujoco.MjModel) -> "Index":
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
        has_flex = m.nflex > 0
        return cls(
            arm_qpos=np.array([m.jnt_qposadr[j] for j in jid]),
            arm_dof=np.array([m.jnt_dofadr[j] for j in jid]),
            arm_ctrl=np.array([[m.actuator_ctrladr[a], m.actuator_ctrladr[a] + 1] for a in aid]),
            tip_site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool_tip"),
            vert_adr=int(m.flex_vertadr[0]) if has_flex else 0,
            vert_num=int(m.flex_vertnum[0]) if has_flex else 0,
        )


def export_xml(path: str | Path, tool: str = "press", clay: ClayCfg = ClayCfg(), sim: SimCfg = SimCfg()) -> Path:
    """Write the compiled scene to MJCF (for MuJoCo Studio).  Mesh paths are absolute."""
    spec = build_spec(tool, clay, sim)
    spec.compile()
    spec.meshdir = str(panda_dir() / "assets")      # the only mesh files are the Panda's; tool heads are inline vertices
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml())
    return path
