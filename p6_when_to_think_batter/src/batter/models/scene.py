"""MJCF scene construction (MuJoCo 4.0): the batting cage.

Two models share the same ball, air, timestep and integrator, so the ball's free
flight is identical in both:

* ``build_flight_model``: the ball alone (flight-model identification and tests);
* ``build_cage_model``: Menagerie Franka Panda (no hand) on a pedestal, holding a thin
  flat bat, plus the ball, the floor, a pitching machine and a target zone.

Panda changes relative to Menagerie (all documented, nothing else is touched):

* the seven position servos become torque motors limited to the Franka datasheet
  torques (87, 87, 87, 87, 12, 12, 12 N m); the controller is model-based;
* a ``bat`` body is welded to the flange (``attachment``): a handle and a flat blade,
  ``bat.thickness`` = 1.2 cm thick.  The blade's face normal is the bat frame's x axis
  and its long axis is the flange axis (bat z).  Site ``sweet`` is the blade centre.

Contacts
* The bat collides with the ball only, through an explicit ``<pair>``.  The pair is
  frictionless (condim 1) and nearly elastic: following the MuJoCo 4.0 notes on
  speculative contacts and restitution, it uses a direct (negative) ``solref`` with zero
  stiffness and damping b = (1 + e) / h, so that the damper turns a closing normal speed
  v into -e v in one step.  ``e`` is ``bat.restitution``.
* The ball collides with the floor and with the arm's collision meshes by the default
  geom filter.

4.0 features: ``<flag speculative>``, a *delayed* ``framepos`` sensor for the camera
(``nsample``/``delay``), contact sensors, a mocap target zone (per-world in ``mujoco.batch``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

# Franka Emika Panda datasheet limits (Franka Control Interface documentation).
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])        # N m
QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])  # rad/s
QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])         # rad/s^2
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
BASE_HEIGHT = 0.55                # top of the pedestal = robot base flange [m]
PARK_POS = (5.0, 0.0, 1.2)        # where the ball sits before a pitch is loaded

CONTACT_FOUND_FORCE = 3           # contact-sensor data bitmask: found | force
CONTACT_REDUCE_NETFORCE = 3

DEFAULT_SIM = {"timestep": 0.010, "integrator": "Euler", "density": 1.2, "viscosity": 1.8e-5, "speculative": True}
DEFAULT_BALL = {"radius": 0.0365, "mass": 0.145}
DEFAULT_BAT = {"thickness": 0.012, "width": 0.09, "length": 0.38, "handle": 0.10, "mass": 0.30, "restitution": 0.55}


@lru_cache(maxsize=1)
def panda_dir() -> Path:
    import mujoco_menagerie as mm

    return Path(mm.prefetch(["franka_emika_panda"], progress=None)[0])


def _set_options(spec: mujoco.MjSpec, sim: dict) -> None:
    o = spec.option
    o.timestep = sim["timestep"]
    o.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + sim["integrator"].upper())
    o.density = sim["density"]
    o.viscosity = sim["viscosity"]
    if sim.get("speculative", True):
        o.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)


def _add_ball(spec: mujoco.MjSpec, ball: dict) -> None:
    body = spec.worldbody.add_body()
    body.name = "ball"
    body.pos = PARK_POS
    body.add_freejoint().name = "ball"
    g = body.add_geom()
    g.name = "ball"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [ball["radius"], 0, 0]
    g.mass = ball["mass"]
    g.rgba = [0.95, 0.95, 0.92, 1]
    g.fluid_ellipsoid = 1          # ellipsoid fluid model: quadratic drag + Magnus lift on the spinning ball
    g.condim = 3
    g.friction = [0.6, 0.005, 0.0001]


def _frame_sensor(spec: mujoco.MjSpec, name: str, typ, body: str, delay: float = 0.0, h: float = 0.01) -> None:
    s = spec.add_sensor()
    s.name = name
    s.type = typ
    s.objtype = mujoco.mjtObj.mjOBJ_BODY
    s.objname = body
    if delay > 0:
        s.nsample = int(round(delay / h))      # delay = nsample * timestep: a plain ring buffer, causal
        s.delay = s.nsample * h


def _contact_sensor(spec: mujoco.MjSpec, name: str, geom1: str, geom2: str) -> None:
    s = spec.add_sensor()
    s.name = name
    s.type = mujoco.mjtSensor.mjSENS_CONTACT
    s.objtype = mujoco.mjtObj.mjOBJ_GEOM
    s.objname = geom1
    s.reftype = mujoco.mjtObj.mjOBJ_GEOM
    s.refname = geom2
    s.intprm[0] = CONTACT_FOUND_FORCE
    s.intprm[1] = CONTACT_REDUCE_NETFORCE
    s.intprm[2] = 1


def build_flight_spec(sim: dict | None = None, ball: dict | None = None) -> mujoco.MjSpec:
    sim, ball = {**DEFAULT_SIM, **(sim or {})}, {**DEFAULT_BALL, **(ball or {})}
    spec = mujoco.MjSpec()
    spec.modelname = "flight"
    _set_options(spec, dict(sim, speculative=False))
    _add_ball(spec, ball)
    return spec


def build_flight_model(sim: dict | None = None, ball: dict | None = None) -> mujoco.MjModel:
    return build_flight_spec(sim, ball).compile()


def _look_at(pos, target) -> list[float]:
    """Quaternion of a MuJoCo camera (looks along -z, y up) at ``pos`` aimed at ``target``."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).flatten())
    return q.tolist()


def _box(parent, name, pos, size, rgba, collide=False):
    g = parent.add_geom()
    if name:
        g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos, g.size, g.rgba = list(pos), list(size), list(rgba)
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


def arm_spec(bat: dict, h: float, kind: str = "blade", paddle: dict | None = None) -> mujoco.MjSpec:
    """Menagerie Panda (no hand) with torque motors, on its base flange at BASE_HEIGHT, holding a bat.

    kind "blade": the batter's flat bat.  kind "paddle": a thin round paddle (rally); its disc
    is a cylinder whose axis is the sweet-spot site's x axis, so the face normal is the same
    task-frame axis as for the blade and the arm's kinematics code is shared.
    """
    spec = mujoco.MjSpec.from_file(str(panda_dir() / "panda_nohand.xml"))
    spec.meshdir = str(panda_dir() / "assets")
    # --- arm actuators: position servos -> torque motors with datasheet limits
    for i in range(7):
        a = spec.actuator(f"actuator{i + 1}")
        a.set_to_motor()
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-TAU_MAX[i], TAU_MAX[i]]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-TAU_MAX[i], TAU_MAX[i]]
    for k in list(spec.keys):
        spec.delete(k)
    spec.body("link0").pos = [0.0, 0.0, BASE_HEIGHT]

    b = spec.body("attachment").add_body()
    b.name = "bat"
    if kind == "blade":
        hl, bl = bat["handle"], bat["length"]
        mass, half_thick, center = bat["mass"], bat["thickness"] / 2, hl + bl / 2
    elif kind == "paddle":
        hl, rad = paddle["handle"], paddle["radius"]
        mass, half_thick, center = paddle["mass"], paddle["thickness"] / 2, paddle["handle"] + paddle["radius"]
    else:
        raise ValueError(kind)
    handle = b.add_geom()
    handle.name = "bat_handle"
    handle.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    handle.size = [0.016, hl / 2, 0]
    handle.pos = [0, 0, hl / 2]
    handle.rgba = [0.15, 0.15, 0.15, 1]
    handle.mass = 0.2 * mass
    handle.contype = 0
    handle.conaffinity = 0
    blade = b.add_geom()
    blade.name = "bat"
    if kind == "blade":
        blade.type = mujoco.mjtGeom.mjGEOM_BOX
        blade.size = [half_thick, bat["width"] / 2, bl / 2]
        blade.rgba = [0.80, 0.62, 0.36, 1]
    else:
        blade.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        blade.size = [rad, half_thick, 0]
        blade.quat = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]     # cylinder axis (z) -> site x
        blade.rgba = [0.75, 0.10, 0.10, 1]
    blade.pos = [0, 0, center]
    blade.mass = 0.8 * mass
    blade.contype = 0             # collides with the ball only, via an explicit pair
    blade.conaffinity = 0
    sweet = b.add_site()
    sweet.name = "sweet"
    sweet.pos = [0, 0, center]
    sweet.size = [0.008, 0.008, 0.008]
    sweet.rgba = [1, 0, 0, 0.5]
    sweet.group = 4
    return spec


def build_cage_spec(cfg: dict | None = None, sim_override: dict | None = None, bat_kind: str = "blade") -> mujoco.MjSpec:
    """Panda + bat + ball + floor + pitching machine + target zone.  ``cfg`` is the YAML config.

    ``bat_kind`` "paddle" swaps the bat for the rally's paddle (``cfg["paddle"]``).
    """
    cfg = cfg or {}
    sim = {**DEFAULT_SIM, **cfg.get("sim", {}), **(sim_override or {})}
    ball = {**DEFAULT_BALL, **cfg.get("ball", {})}
    bat = {**DEFAULT_BAT, **cfg.get("bat", {})}
    h = sim["timestep"]
    spec = arm_spec(bat, h, kind=bat_kind, paddle=cfg.get("paddle"))
    spec.modelname = "batting_cage"
    _set_options(spec, sim)

    # --- environment
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    getattr(spec.visual, "global").offwidth = 1920
    getattr(spec.visual, "global").offheight = 1080
    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2 = [0.62, 0.76, 0.93], [0.93, 0.96, 1.0]
    sky.width, sky.height = 256, 1536
    wb = spec.worldbody
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.24, 0.42, 0.26, 1]
    _box(wb, "pedestal", (-0.03, 0, BASE_HEIGHT / 2), (0.14, 0.14, BASE_HEIGHT / 2), (0.35, 0.37, 0.40, 1))
    _box(wb, "plate", (0.62, 0, 0.003), (0.15, 0.15, 0.003), (0.92, 0.92, 0.92, 1))
    px = cfg.get("pitch", {}).get("release_x", 5.0)
    _box(wb, "machine", (px + 0.35, 0, 0.6), (0.25, 0.2, 0.6), (0.55, 0.20, 0.18, 1))
    _box(wb, "machine_barrel", (px + 0.12, 0, 1.22), (0.12, 0.06, 0.28), (0.2, 0.2, 0.2, 1))
    light = wb.add_light()
    light.pos = [2.0, -2.0, 4.0]
    light.dir = [-0.2, 0.3, -1]
    light2 = wb.add_light()
    light2.pos = [3.0, 2.5, 3.0]
    light2.dir = [0, -0.5, -1]

    # target zone: a mocap body, so each world of a batch can have its own target
    tb = wb.add_body()
    tb.name = "target"
    tb.mocap = True
    tb.pos = [6.0, 0.0, 0.004]
    tg = tb.add_geom()
    tg.name = "target"
    tg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    tg.size = [cfg.get("target", {}).get("radius", 0.75), 0.002, 0]
    tg.rgba = [0.95, 0.75, 0.10, 0.9]
    tg.contype = 0
    tg.conaffinity = 0

    for name, pos, look, fovy in [("side", (1.6, -3.4, 1.3), (1.9, 0.0, 0.9), 55),
                                  ("close", (1.5, -1.3, 1.25), (0.55, 0.0, 0.95), 45),
                                  ("behind", (-1.6, -0.5, 1.7), (2.5, 0.0, 0.8), 55)]:
        cam = wb.add_camera()
        cam.name = name
        cam.pos = list(pos)
        cam.quat = _look_at(pos, look)
        cam.fovy = fovy

    _add_ball(spec, ball)

    # --- bat-ball contact: frictionless, restitution e (see the module docstring)
    p = spec.add_pair()
    p.name = "bat_ball"
    p.geomname1 = "bat"
    p.geomname2 = "ball"
    p.condim = 1
    p.solref = [0.0, -(1.0 + bat["restitution"]) / h]
    p.solimp = [0.995, 0.995, 0.001, 0.5, 2]

    # --- sensors
    delay = cfg.get("measurement", {}).get("delay", 0.03)
    _frame_sensor(spec, "ball_cam", mujoco.mjtSensor.mjSENS_FRAMEPOS, "ball", delay=delay, h=h)   # camera (delayed)
    _frame_sensor(spec, "ball_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS, "ball")                    # ground truth
    _frame_sensor(spec, "ball_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL, "ball")
    _contact_sensor(spec, "hit", "bat", "ball")
    _contact_sensor(spec, "ground", "floor", "ball")
    return spec


@dataclass
class CageIndex:
    """Addresses into qpos/qvel/ctrl/sensordata for the cage model."""

    arm_qadr: np.ndarray
    arm_dadr: np.ndarray
    ball_qadr: int
    ball_dadr: int
    ball_body: int
    bat_body: int
    bat_geom: int
    ball_geom: int
    sweet_site: int
    target_mocap: int
    sensors: dict[str, slice]

    @staticmethod
    def of(m: mujoco.MjModel) -> "CageIndex":
        jid = [m.joint(j).id for j in ARM_JOINTS]
        bj = m.joint("ball").id
        sens = {m.sensor(i).name: slice(int(m.sensor_adr[i]), int(m.sensor_adr[i] + m.sensor_dim[i]))
                for i in range(m.nsensor)}
        return CageIndex(
            arm_qadr=m.jnt_qposadr[jid].copy(), arm_dadr=m.jnt_dofadr[jid].copy(),
            ball_qadr=int(m.jnt_qposadr[bj]), ball_dadr=int(m.jnt_dofadr[bj]),
            ball_body=m.body("ball").id, bat_body=m.body("bat").id, bat_geom=m.geom("bat").id,
            ball_geom=m.geom("ball").id, sweet_site=m.site("sweet").id,
            target_mocap=int(m.body_mocapid[m.body("target").id]), sensors=sens)


def build_cage_model(cfg: dict | None = None, sim_override: dict | None = None) -> mujoco.MjModel:
    return build_cage_spec(cfg, sim_override).compile()


def export_xml(path: Path, cfg: dict | None = None) -> Path:
    """Write a standalone MJCF (absolute mesh paths) for MuJoCo Studio:
    ``python -m mujoco.studio.viewer --model=<path> --hot_reload=all``."""
    spec = build_cage_spec(cfg)
    spec.compile()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml())
    return path
