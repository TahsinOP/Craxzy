"""The rally scene: two Pandas with paddles, a regulation table and net, a ping-pong ball.

* Arm A is the batter's arm (base at the origin, facing +x), arm B is a second copy attached
  with the prefix ``b_`` at x = L, rotated by pi about the vertical.
* Each paddle collides with the ball only, through a frictionless pair with a direct
  restitution law (as the bat in the batting cage).
* The table collides with the ball only, through a frictionless pair with the same direct
  restitution law.  (With friction, measured on simulated bounces, Coulomb's law held but the
  ball could leave the table up to four times faster than it arrived.)  The net collides with the ball through the default contact.  The arms do not collide with the
  table (they cannot reach the playing surface from 0.75 m behind its ends).
* Sensors: the delayed camera (``ball_cam``), the true ball state, and contact sensors for
  each paddle, the table, the net and the floor.

``build_arm_model`` is the single-arm version (the batting cage's layout with the paddle) that
both arms' planners use for kinematics, in their own frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from batter.models.scene import (BASE_HEIGHT, DEFAULT_BALL, DEFAULT_SIM, _add_ball, _box, _contact_sensor, _frame_sensor,
                                 _look_at, _set_options, arm_spec, build_cage_spec)

ARM_JOINTS_B = [f"b_joint{i}" for i in range(1, 8)]


def _quat_z(angle: float) -> list[float]:
    return [float(np.cos(angle / 2)), 0.0, 0.0, float(np.sin(angle / 2))]


def build_arm_model(cfg: dict) -> mujoco.MjModel:
    """One arm with the paddle, in the batting cage's layout: kinematics for either arm's planner."""
    return build_cage_spec(cfg, bat_kind="paddle").compile()


def build_rally_spec(cfg: dict, sim_override: dict | None = None) -> mujoco.MjSpec:
    sim = {**DEFAULT_SIM, **cfg.get("sim", {}), **(sim_override or {})}
    ball = {**DEFAULT_BALL, **cfg["ball"]}
    t, pd = cfg["table"], cfg["paddle"]
    h = sim["timestep"]
    L = t["L"]

    spec = arm_spec(None, h, kind="paddle", paddle=pd)          # arm A: names unprefixed
    spec.modelname = "rally"
    _set_options(spec, sim)
    child = arm_spec(None, h, kind="paddle", paddle=pd)         # arm B
    _set_options(child, sim)                                    # same integrator: no attach conflict
    frame = spec.worldbody.add_frame()
    frame.pos = [L, 0.0, 0.0]
    frame.quat = _quat_z(np.pi)
    frame.attach_body(child.body("link0"), "b_", "")

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
    floor.rgba = [0.42, 0.36, 0.30, 1]
    for x, name in [(-0.03, "pedestal"), (L + 0.03, "b_pedestal")]:
        _box(wb, name, (x, 0, BASE_HEIGHT / 2), (0.14, 0.14, BASE_HEIGHT / 2), (0.35, 0.37, 0.40, 1))

    # table: top surface at t["height"], collides with the ball only (pair below)
    x0, x1 = t["gap"], t["gap"] + t["length"]
    xm = 0.5 * (x0 + x1)
    top = _box(wb, "table", (xm, 0, t["height"] - 0.015), (t["length"] / 2, t["width"] / 2, 0.015), (0.10, 0.27, 0.52, 1))
    for x in (x0 + 0.1, x1 - 0.1):
        for y in (-t["width"] / 2 + 0.1, t["width"] / 2 - 0.1):
            _box(wb, None, (x, y, (t["height"] - 0.03) / 2), (0.025, 0.025, (t["height"] - 0.03) / 2), (0.2, 0.2, 0.2, 1))
    _box(wb, "centre_line", (xm, 0, t["height"] + 0.0005), (t["length"] / 2, 0.0015, 0.0005), (0.95, 0.95, 0.95, 1))
    for x in (x0 + 0.01, x1 - 0.01):
        _box(wb, None, (x, 0, t["height"] + 0.0005), (0.01, t["width"] / 2, 0.0005), (0.95, 0.95, 0.95, 1))
    half_net = t["width"] / 2 + t["net_overhang"]
    net = _box(wb, "net", (xm, 0, t["height"] + t["net_height"] / 2), (0.004, half_net, t["net_height"] / 2),
               (0.92, 0.92, 0.92, 0.85), collide=True)
    net.contype, net.conaffinity = 1, 1

    # landing targets (visual): where each arm aims on the far half
    for name, x in [("target", x1 - 0.7), ("b_target", x0 + 0.7)]:
        tb = wb.add_body()
        tb.name = name
        tb.mocap = True
        tb.pos = [x, 0.0, t["height"] + 0.001]
        tg = tb.add_geom()
        tg.name = name
        tg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        tg.size = [0.06, 0.0008, 0]
        tg.rgba = [0.95, 0.75, 0.10, 0.9]
        tg.contype = tg.conaffinity = 0

    light = wb.add_light()
    light.pos = [xm, -1.5, 3.5]
    light.dir = [0, 0.3, -1]
    light2 = wb.add_light()
    light2.pos = [xm, 1.5, 3.0]
    light2.dir = [0, -0.4, -1]
    for name, pos, look, fovy in [("side", (xm, -3.6, 1.9), (xm, 0.0, 0.85), 50),
                                  ("behind_a", (-1.3, -0.6, 1.8), (xm, 0.0, 0.8), 55),
                                  ("high", (xm, -2.2, 3.4), (xm, 0.0, 0.8), 55)]:
        cam = wb.add_camera()
        cam.name = name
        cam.pos = list(pos)
        cam.quat = _look_at(pos, look)
        cam.fovy = fovy

    _add_ball(spec, ball)
    g = spec.geom("ball")
    g.rgba = [1.0, 0.55, 0.10, 1]                     # orange ball

    # --- contacts
    for bat in ("bat", "b_bat"):
        p = spec.add_pair()
        p.name = f"{bat}_ball"
        p.geomname1, p.geomname2 = bat, "ball"
        p.condim = 1
        p.solref = [0.0, -(1.0 + pd["restitution"]) / h]
        p.solimp = [0.995, 0.995, 0.001, 0.5, 2]
    p = spec.add_pair()
    p.name = "table_ball"
    p.geomname1, p.geomname2 = "table", "ball"
    p.condim = 1          # frictionless: with friction, this one-step restitution contact can add energy (see rally/flight.py)
    p.solref = [0.0, -(1.0 + t["restitution"]) / h]
    p.solimp = [0.995, 0.995, 0.001, 0.5, 2]

    # --- sensors
    delay = cfg["measurement"]["delay"]
    _frame_sensor(spec, "ball_cam", mujoco.mjtSensor.mjSENS_FRAMEPOS, "ball", delay=delay, h=h)
    _frame_sensor(spec, "ball_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS, "ball")
    _frame_sensor(spec, "ball_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL, "ball")
    for name, geom in [("hit_a", "bat"), ("hit_b", "b_bat"), ("table", "table"), ("net", "net"), ("ground", "floor")]:
        _contact_sensor(spec, name, geom, "ball")
    return spec


def build_rally_model(cfg: dict, sim_override: dict | None = None) -> mujoco.MjModel:
    return build_rally_spec(cfg, sim_override).compile()


@dataclass
class RallyIndex:
    """Addresses into the rally model, per arm (0 = A, 1 = B)."""

    qadr: list[np.ndarray]
    dadr: list[np.ndarray]
    act: list[np.ndarray]
    ball_qadr: int
    ball_dadr: int
    ball_body: int
    sweet: list[int]
    target_mocap: list[int]
    sensors: dict[str, slice]

    @staticmethod
    def of(m: mujoco.MjModel) -> "RallyIndex":
        names = [[f"joint{i}" for i in range(1, 8)], ARM_JOINTS_B]
        jid = [[m.joint(j).id for j in js] for js in names]
        bj = m.joint("ball").id
        sens = {m.sensor(i).name: slice(int(m.sensor_adr[i]), int(m.sensor_adr[i] + m.sensor_dim[i])) for i in range(m.nsensor)}
        return RallyIndex(
            qadr=[m.jnt_qposadr[j].copy() for j in jid], dadr=[m.jnt_dofadr[j].copy() for j in jid],
            act=[np.array([m.actuator(f"{p}actuator{i}").id for i in range(1, 8)]) for p in ("", "b_")],
            ball_qadr=int(m.jnt_qposadr[bj]), ball_dadr=int(m.jnt_dofadr[bj]), ball_body=m.body("ball").id,
            sweet=[m.site("sweet").id, m.site("b_sweet").id],
            target_mocap=[int(m.body_mocapid[m.body(n).id]) for n in ("target", "b_target")], sensors=sens)


# ---------------------------------------------------------------------------- frames
def to_arm_frame(x: np.ndarray, arm: int, L: float) -> np.ndarray:
    """World ball state(s) (..., 9) [or positions (..., 3)] -> the frame of ``arm``.

    Arm B's frame is the world rotated by pi about the vertical through x = L/2: (x, y, z) -> (L - x, -y, z);
    velocities and spins rotate the same way (a proper rotation).  The map is its own inverse.
    """
    if arm == 0:
        return x
    y = np.array(x, float, copy=True)
    n = y.shape[-1] // 3
    for k in range(n):
        y[..., 3 * k] *= -1.0
        y[..., 3 * k + 1] *= -1.0
    y[..., 0] += L
    return y


def cov_to_arm_frame(P: np.ndarray, arm: int) -> np.ndarray:
    """Covariances (..., 3k, 3k) -> the frame of ``arm``."""
    if arm == 0:
        return P
    k = P.shape[-1] // 3
    d = np.tile([-1.0, -1.0, 1.0], k)
    return P * d[:, None] * d[None, :]
