"""Offline videos with MuJoCo 4.0's Filament renderer (stretch goal: nice-looking videos).

Filament is the physically based renderer behind MuJoCo Studio.  It renders offscreen into
a numpy buffer through ``mujoco.rendering.filament.renderer.Renderer``; the scene is kept
in sync with an ``MjData`` by Studio's own ``native_viewer_cc.ModelScene``, which also takes
extra decorative geoms, so the forecast and trail overlays of the classic recorder carry over.

The look is a separate model: the cage's geometry with physically based materials (rough
grass with a mown checker, wood, steel, leather, paint), a sky-blue clear colour and brighter
fallback lighting set through ``filament.*`` custom numerics.  Only materials, textures and
numerics are added, so the look model has the same state sizes as the simulated one and a
world's state is copied across for each frame.

Backends: Vulkan by default (on the development machine the Intel GPU; about 25 ms per
960x540 frame).  ``FilamentRecorder`` raises ``RuntimeError`` if no context can be created;
``render.make_recorder`` then falls back to the classic renderer.
"""
from __future__ import annotations

import mujoco
import numpy as np

from batter.eval.render import label
from batter.models.scene import build_cage_spec

MATERIALS = {
    # name: (rgba, roughness, metallic)
    "grass": ([0.21, 0.43, 0.22, 1.0], 0.95, 0.0),
    "wood": ([0.80, 0.60, 0.34, 1.0], 0.55, 0.0),
    "steel": ([0.55, 0.57, 0.60, 1.0], 0.35, 0.9),
    "leather": ([0.96, 0.95, 0.91, 1.0], 0.6, 0.0),
    "paint": ([0.70, 0.18, 0.15, 1.0], 0.45, 0.2),
    "rubber": ([0.12, 0.12, 0.12, 1.0], 0.8, 0.0),
    "gold": ([0.95, 0.75, 0.10, 1.0], 0.5, 0.0),
    "chalk": ([0.93, 0.93, 0.93, 1.0], 0.9, 0.0),
}
GEOM_MATERIAL = {"floor": "grass", "bat": "wood", "bat_handle": "rubber", "pedestal": "steel", "ball": "leather",
                 "machine": "paint", "machine_barrel": "rubber", "target": "gold", "plate": "chalk"}


def look_model(cfg: dict, sim_override: dict | None = None) -> mujoco.MjModel:
    """The cage with physically based materials and Filament lighting flags."""
    spec = build_cage_spec(cfg, sim_override)
    spec.add_numeric(name="filament.clear_color", data=[0.62, 0.76, 0.93, 1.0])
    spec.add_numeric(name="filament.fallback.environment_light_intensity", data=[20000.0])
    spec.add_numeric(name="filament.fallback.scene_light_intensity", data=[150000.0])
    tex = spec.add_texture()
    tex.name = "mown"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2 = [0.24, 0.47, 0.25], [0.20, 0.41, 0.21]
    tex.width = tex.height = 512
    for name, (rgba, rough, metal) in MATERIALS.items():
        mt = spec.add_material()
        mt.name, mt.rgba, mt.roughness, mt.metallic = name, rgba, rough, metal
        if name == "grass":
            mt.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "mown"
            mt.texrepeat = [1.5, 1.5]
            mt.texuniform = True
    for geom, mat in GEOM_MATERIAL.items():
        spec.geom(geom).material = mat
        spec.geom(geom).rgba = MATERIALS[mat][0]
    return spec.compile()


def _sphere(pos, radius, rgba) -> mujoco.MjvGeom:
    g = mujoco.MjvGeom()
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([radius, 0.0, 0.0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    return g


def _line(a, b, width, rgba) -> mujoco.MjvGeom:
    g = mujoco.MjvGeom()
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), width, np.asarray(a, float), np.asarray(b, float))
    return g


class FilamentRecorder:
    """Drop-in replacement for ``render.Recorder`` that renders with Filament."""

    def __init__(self, env, camera, width: int = 960, height: int = 540, world: int = 0, every: int = 1,
                 text: str | None = None, show_forecast: bool = True, repeat: int = 1):
        from mujoco import mjrf
        from mujoco.rendering.filament.renderer import Renderer
        from mujoco.studio import native_viewer_cc, window  # noqa: F401  (window registers Filament's resource providers)

        self.env, self.world, self.every, self.text = env, world, every, text
        self.show_forecast, self.repeat = show_forecast, repeat
        self.W, self.H = width, height
        sim = {"timestep": env.model.opt.timestep,
               "speculative": bool(env.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE))}
        self.model = look_model(env.cfg, sim)
        if self.model.nq != env.model.nq or self.model.nmocap != env.model.nmocap:
            raise RuntimeError("look model does not match the simulated model")
        self.data = mujoco.MjData(self.model)
        self._sim_data = mujoco.MjData(env.model)
        cfg = mjrf.ContextConfig()
        cfg.graphics_api = mjrf.GraphicsApi.GRAPHICS_API_VULKAN
        try:
            self._ctx = mjrf.Context(cfg)
        except Exception as e:          # no Vulkan device: let the caller fall back
            raise RuntimeError(f"no Filament context: {e}") from e
        self._renderer = Renderer(self._ctx)
        self._scene = native_viewer_cc.ModelScene(self._renderer.context, self._renderer.scene("scn"), self.model)
        self._buf = np.zeros((height, width, 3), np.uint8)
        self._renderer.target("out", self._buf)
        if isinstance(camera, str):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = self.model.camera(camera).id
            camera = cam
        self.cam = camera
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        self.frames: list[np.ndarray] = []
        self.pitches: list[list[np.ndarray]] = []
        self.trail: list[np.ndarray] = []
        self.calls: list[np.ndarray] = []
        self._n_plans = 0

    def new_pitch(self) -> None:
        self.trail, self.calls, self._n_plans = [], [], 0
        self.pitches.append([])

    def _overlays(self, env) -> list:
        i = self.world
        g = [_sphere(p, 0.012, (1.0, 1.0, 1.0, 0.6)) for p in self.trail[-60:]]
        if self.show_forecast and env.ekf.started[i] and not env.hit[i]:
            path = env.traj[env.nd::2, i, 0:3]
            g += [_line(a, b, 0.006, (0.11, 0.69, 0.48, 1.0)) for a, b in zip(path[:-1], path[1:]) if a[2] > 0.03 and a[0] > 0.3]
            n = int(env.plan_len[i])
            if env.plans[i] is not None and n > 2:
                path = env.plan_p[i, :n:2]
                g += [_line(a, b, 0.004, (0.92, 0.41, 0.20, 1.0)) for a, b in zip(path[:-1], path[1:]) if a[2] > 0.03 and a[0] > 0.3]
        g += [_sphere(p, 0.022, (0.93, 0.63, 0.0, 1.0)) for p in self.calls]
        return g

    def __call__(self, env) -> None:
        from mujoco.rendering.filament.renderer import View

        i = self.world
        if env.k == 0:
            self.new_pitch()
        ball = env.ball_state()[i, 0:3].copy()
        self.trail.append(ball)
        if env.n_plans[i] > self._n_plans:
            self._n_plans = int(env.n_plans[i])
            self.calls.append(ball)
        if env.k % self.every:
            return
        env.batch.get_data(i, self._sim_data)
        d = self.data
        d.qpos[:] = self._sim_data.qpos
        d.qvel[:] = self._sim_data.qvel
        d.mocap_pos[:] = self._sim_data.mocap_pos
        d.mocap_quat[:] = self._sim_data.mocap_quat
        d.time = self._sim_data.time
        mujoco.mj_forward(self.model, d)
        self._scene.sync(self.W, self.H, self.model, d, self.pert, self.cam, self.opt, self._overlays(env))
        self._renderer.render([View(scene="scn", target="out", camera=mujoco.mjv_camera2GLCamera(self.model, d, self.cam),
                                    headlight=True, headlight_intensity=0.5 * self._scene.get_headlight_intensity())])
        img = self._buf.copy()
        if self.text:
            img = label(img, self.text)
        img = label(img, f"t = {env.k * env.h * 1e3:4.0f} ms   planner calls {int(env.n_plans[i])}", size=16,
                    xy=(10, img.shape[0] - 30))
        self.frames.extend([img] * self.repeat)
        self.pitches[-1].extend([img] * self.repeat)

    def close(self) -> None:
        del self._scene, self._renderer, self._ctx
