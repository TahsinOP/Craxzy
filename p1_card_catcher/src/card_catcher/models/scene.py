"""MJCF scene construction (MuJoCo 4.0).

Two models share the *same* object, fluid medium, timestep and integrator, so the
object's free flight is identical in both until first contact:

* ``build_flight_model``: the object alone (fast; used for datasets, M0-M2).
* ``build_catch_model``: Menagerie Franka Panda on a table + the object (M0, M3, M4).

Panda changes relative to Menagerie (all documented, nothing else is touched):

* the seven position servos are replaced by torque motors limited to the Franka
  datasheet torques (87, 87, 87, 87, 12, 12, 12 N m); the controller in
  ``card_catcher.control`` is model-based and commands torque;
* the finger actuator is kept (position servo on the ``split`` tendon, ctrl 0..255);
* a ``tcp`` site is added at the centre of the finger pads;
* the base is mounted *inverted* on a gantry at height ``BASE_HEIGHT`` above the
  floor (rotation of pi about the world x-axis).  Reason (measured, see
  results/M0.md): the wrist pitch of the Panda's TCP axis is
  q6 + q4 - q2 - pi/2 in the base frame, with q6 <= 3.75 rad, so an upright
  table-mounted Panda can face its palm forward-and-up -- towards a descending
  object -- only at the very top of its workspace and within 0.1-0.2 rad of the
  joint-6 limit.  Inverted, the same palm directions sit in the middle of the
  wrist range.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from card_catcher.models.objects import OBJECTS, ObjectSpec

# Franka Emika Panda datasheet limits (Franka Control Interface documentation).
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])        # N m
QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])  # rad/s
QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])         # rad/s^2
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
TCP_OFFSET = (0.0, 0.0, 0.1034)   # hand frame -> centre of the fingertip pads
FLOOR_Z = 0.0                     # floor
TABLE_TOP = 0.25                  # low table under the catch region [m]
BASE_HEIGHT = 1.30                # height of the (inverted) robot base flange [m]
PARK_POS = (3.0, 0.0, 2.5)        # where the object sits before a throw is loaded

CONTACT_FOUND_FORCE = 3           # contact-sensor data bitmask: found | force
CONTACT_REDUCE_NETFORCE = 3


@lru_cache(maxsize=1)
def panda_dir() -> Path:
    import mujoco_menagerie as mm

    return Path(mm.prefetch(["franka_emika_panda"], progress=None)[0])


def _option_xml(dt: float, integrator: str, density: float, viscosity: float, speculative: bool) -> str:
    flag = '<flag speculative="enable"/>' if speculative else ""
    return (
        f'<option timestep="{dt:g}" integrator="{integrator}" density="{density:g}" '
        f'viscosity="{viscosity:g}">{flag}</option>'
    )


def flight_xml(obj: ObjectSpec, dt: float = 0.008, integrator: str = "implicitfast",
               density: float = 1.2, viscosity: float = 1.8e-5) -> str:
    return f"""<mujoco model="flight_{obj.name}">
  {_option_xml(dt, integrator, density, viscosity, False)}
  <worldbody>
    <body name="obj" pos="0 0 1"><freejoint name="obj"/>{obj.geom_xml()}</body>
  </worldbody>
</mujoco>"""


def build_flight_model(obj: str | ObjectSpec, sim: dict | None = None) -> mujoco.MjModel:
    """Object-only model.  ``sim`` is the ``sim`` section of the YAML config."""
    spec = OBJECTS[obj] if isinstance(obj, str) else obj
    sim = sim or {}
    return mujoco.MjModel.from_xml_string(
        flight_xml(spec, sim.get("timestep", 0.008), sim.get("integrator", "implicitfast"),
                   sim.get("density", 1.2), sim.get("viscosity", 1.8e-5))
    )


def _add_contact_sensor(spec: mujoco.MjSpec, name: str, objtype, objname: str, refname: str) -> None:
    s = spec.add_sensor()
    s.name = name
    s.type = mujoco.mjtSensor.mjSENS_CONTACT
    s.objtype = objtype
    s.objname = objname
    s.reftype = mujoco.mjtObj.mjOBJ_BODY
    s.refname = refname
    s.intprm[0] = CONTACT_FOUND_FORCE
    s.intprm[1] = CONTACT_REDUCE_NETFORCE
    s.intprm[2] = 1


def build_catch_spec(obj: str | ObjectSpec, sim: dict | None = None, effector: str = "gripper",
                     suction_force: float = 0.0, mount: str = "inverted") -> mujoco.MjSpec:
    """Panda + table + floor + thrown object.

    effector: "gripper" (parallel jaws) or "suction" (stretch goal: a suction disc
    on the palm with the 4.0 ``adhesion`` geom attribute, ``suction_force`` newtons
    per contact).
    """
    ospec = OBJECTS[obj] if isinstance(obj, str) else obj
    sim = sim or {}
    spec = mujoco.MjSpec.from_file(str(panda_dir() / "panda.xml"))
    spec.modelname = f"catch_{ospec.name}"
    spec.meshdir = str(panda_dir() / "assets")

    o = spec.option
    o.timestep = sim.get("timestep", 0.008)
    o.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + sim.get("integrator", "implicitfast").upper())
    o.density = sim.get("density", 1.2)
    o.viscosity = sim.get("viscosity", 1.8e-5)
    if sim.get("speculative", True):
        o.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)

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

    base = spec.body("link0")
    if mount == "inverted":
        base.pos = [0.0, 0.0, BASE_HEIGHT]
        base.quat = [0.0, 1.0, 0.0, 0.0]      # rotation of pi about world x: inverted mount
    elif mount == "upright":                  # only used by scripts/m0_mount.py (reachability comparison)
        base.pos = [0.0, 0.0, TABLE_TOP]
    else:
        raise ValueError(mount)

    hand = spec.body("hand")
    tcp = hand.add_site()
    tcp.name = "tcp"
    tcp.pos = TCP_OFFSET
    tcp.size = [0.006, 0.006, 0.006]
    tcp.rgba = [1, 0, 0, 0.6]
    tcp.group = 4

    if effector == "suction":
        # Suction disc filling the jaw opening, flush with the fingertips.
        g = hand.add_geom()
        g.name = "suction_cup"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [0.035, 0.004, 0]
        g.pos = [0, 0, TCP_OFFSET[2] + 0.006]
        g.rgba = [0.1, 0.1, 0.1, 1]
        g.adhesion = suction_force
        g.mass = 0.02
    elif effector != "gripper":
        raise ValueError(effector)

    # --- environment
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    wb = spec.worldbody
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.pos = [0, 0, FLOOR_Z]
    floor.rgba = [0.25, 0.28, 0.32, 1]
    table = wb.add_geom()
    table.name = "table"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [0.45, 0.50, 0.02]
    table.pos = [0.45, 0, TABLE_TOP - 0.02]
    table.rgba = [0.55, 0.45, 0.35, 1]
    for x, y in [(0.05, -0.45), (0.05, 0.45), (0.85, -0.45), (0.85, 0.45)]:
        leg = wb.add_geom()
        leg.type = mujoco.mjtGeom.mjGEOM_BOX
        leg.size = [0.02, 0.02, (TABLE_TOP - 0.04) / 2]
        leg.pos = [x, y, (TABLE_TOP - 0.04) / 2]
        leg.rgba = [0.4, 0.33, 0.26, 1]
        leg.contype = 0
        leg.conaffinity = 0
    # gantry carrying the inverted arm (visual only; it is behind and above the workspace)
    for pos, size in [((-0.10, 0.0, BASE_HEIGHT + 0.03), (0.12, 0.75, 0.03)),
                      ((-0.10, -0.72, (BASE_HEIGHT + 0.06) / 2), (0.03, 0.03, (BASE_HEIGHT + 0.06) / 2)),
                      ((-0.10, 0.72, (BASE_HEIGHT + 0.06) / 2), (0.03, 0.03, (BASE_HEIGHT + 0.06) / 2))]:
        beam = wb.add_geom()
        beam.type = mujoco.mjtGeom.mjGEOM_BOX
        beam.size = list(size)
        beam.pos = list(pos)
        beam.rgba = [0.35, 0.37, 0.40, 1]
        beam.contype = 0
        beam.conaffinity = 0
    light = wb.add_light()
    light.pos = [1.0, -1.0, 2.5]
    light.dir = [-0.3, 0.3, -1]
    light2 = wb.add_light()
    light2.pos = [0.5, 1.5, 2.0]
    light2.dir = [0, -0.5, -1]

    cam = wb.add_camera()
    cam.name = "side"
    cam.pos = [1.3, -2.6, 0.9]
    cam.quat = _look_at(cam.pos, [1.1, 0.0, 0.55])
    cam.fovy = 50
    cam2 = wb.add_camera()
    cam2.name = "close"
    cam2.pos = [0.75, -0.75, 0.75]
    cam2.quat = _look_at(cam2.pos, [0.55, 0.0, 0.45])
    cam2.fovy = 45

    # --- thrown object
    body = wb.add_body()
    body.name = "obj"
    body.pos = PARK_POS
    body.add_freejoint().name = "obj"
    g = body.add_geom()
    g.name = "obj"
    g.type = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "box": mujoco.mjtGeom.mjGEOM_BOX}[ospec.gtype]
    g.size = list(ospec.size) + [0.0] * (3 - len(ospec.size))
    g.mass = ospec.mass
    g.rgba = ospec.rgba
    g.fluid_ellipsoid = 1
    g.fluid_coefs = list(ospec.fluidcoef)
    g.friction = list(ospec.friction)
    g.condim = ospec.condim

    # --- sensors: ground truth + 4.0 contact sensors
    for name, typ in [("obj_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS), ("obj_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL)]:
        s = spec.add_sensor()
        s.name = name
        s.type = typ
        s.objtype = mujoco.mjtObj.mjOBJ_BODY
        s.objname = "obj"
    B = mujoco.mjtObj.mjOBJ_BODY
    _add_contact_sensor(spec, "touch_left", B, "left_finger", "obj")
    _add_contact_sensor(spec, "touch_right", B, "right_finger", "obj")
    _add_contact_sensor(spec, "touch_hand", B, "hand", "obj")
    _add_contact_sensor(spec, "touch_floor", mujoco.mjtObj.mjOBJ_GEOM, "floor", "obj")
    _add_contact_sensor(spec, "touch_table", mujoco.mjtObj.mjOBJ_GEOM, "table", "obj")
    return spec


def _look_at(pos, target) -> list[float]:
    """Quaternion of a MuJoCo camera (looks along -z, y up) at ``pos`` aimed at ``target``."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q.tolist()


@dataclass
class CatchIndex:
    """Addresses into qpos/qvel/ctrl/sensordata for the catch model."""

    arm_qadr: np.ndarray
    arm_dadr: np.ndarray
    finger_qadr: np.ndarray
    obj_qadr: int
    obj_dadr: int
    obj_body: int
    hand_body: int
    tcp_site: int
    grip_act: int
    sensors: dict[str, slice]

    @staticmethod
    def of(m: mujoco.MjModel) -> "CatchIndex":
        jid = [m.joint(j).id for j in ARM_JOINTS]
        oj = m.joint("obj").id
        sens = {}
        for i in range(m.nsensor):
            sens[m.sensor(i).name] = slice(int(m.sensor_adr[i]), int(m.sensor_adr[i] + m.sensor_dim[i]))
        return CatchIndex(
            arm_qadr=m.jnt_qposadr[jid].copy(),
            arm_dadr=m.jnt_dofadr[jid].copy(),
            finger_qadr=np.array([m.jnt_qposadr[m.joint(j).id] for j in ("finger_joint1", "finger_joint2")]),
            obj_qadr=int(m.jnt_qposadr[oj]),
            obj_dadr=int(m.jnt_dofadr[oj]),
            obj_body=m.body("obj").id,
            hand_body=m.body("hand").id,
            tcp_site=m.site("tcp").id,
            grip_act=m.actuator("actuator8").id,
            sensors=sens,
        )


def build_catch_model(obj: str | ObjectSpec, sim: dict | None = None, **kw) -> mujoco.MjModel:
    return build_catch_spec(obj, sim, **kw).compile()


def export_xml(obj: str, path: Path, sim: dict | None = None, **kw) -> Path:
    """Write a standalone MJCF (absolute mesh paths) for MuJoCo Studio:
    ``python -m mujoco.studio.viewer --model=<path> --hot_reload=all``."""
    spec = build_catch_spec(obj, sim, **kw)
    spec.compile()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml())
    return path
