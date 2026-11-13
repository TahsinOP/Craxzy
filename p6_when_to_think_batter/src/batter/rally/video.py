"""Filament video of a rally: the rally scene with physically based materials, the ball's trail
and the filter's forecast (green) drawn as overlays, and a marker wherever a planner ran."""
from __future__ import annotations

import mujoco
import numpy as np

from batter.eval.filament import _line, _sphere
from batter.eval.render import label
from batter.rally.scene import build_rally_spec

MATERIALS = {"table": ([0.08, 0.24, 0.48, 1.0], 0.55, 0.0), "net": ([0.93, 0.93, 0.93, 1.0], 0.8, 0.0),
             "paddle": ([0.78, 0.08, 0.08, 1.0], 0.7, 0.0), "rubber": ([0.12, 0.12, 0.12, 1.0], 0.8, 0.0),
             "steel": ([0.55, 0.57, 0.60, 1.0], 0.35, 0.9), "floor": ([0.48, 0.38, 0.28, 1.0], 0.6, 0.0),
             "ball": ([1.0, 0.55, 0.10, 1.0], 0.5, 0.0), "chalk": ([0.95, 0.95, 0.95, 1.0], 0.9, 0.0),
             "gold": ([0.95, 0.75, 0.10, 1.0], 0.5, 0.0)}
GEOMS = {"table": "table", "net": "net", "bat": "paddle", "b_bat": "paddle", "bat_handle": "rubber", "b_bat_handle": "rubber",
         "pedestal": "steel", "b_pedestal": "steel", "floor": "floor", "ball": "ball", "centre_line": "chalk",
         "target": "gold", "b_target": "gold"}


def look_model(cfg: dict) -> mujoco.MjModel:
    spec = build_rally_spec(cfg)
    spec.add_numeric(name="filament.clear_color", data=[0.62, 0.76, 0.93, 1.0])
    spec.add_numeric(name="filament.fallback.environment_light_intensity", data=[20000.0])
    spec.add_numeric(name="filament.fallback.scene_light_intensity", data=[150000.0])
    tex = spec.add_texture()
    tex.name = "boards"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2 = [0.50, 0.40, 0.30], [0.44, 0.35, 0.26]
    tex.width = tex.height = 512
    for name, (rgba, rough, metal) in MATERIALS.items():
        mt = spec.add_material()
        mt.name, mt.rgba, mt.roughness, mt.metallic = name, rgba, rough, metal
        if name == "floor":
            mt.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "boards"
            mt.texrepeat = [2.0, 2.0]
            mt.texuniform = True
    for geom, mat in GEOMS.items():
        spec.geom(geom).material = mat
        spec.geom(geom).rgba = MATERIALS[mat][0]
    return spec.compile()


class RallyRecorder:
    def __init__(self, env, camera, width: int = 1280, height: int = 720, text: str | None = None, repeat: int = 1):
        from mujoco import mjrf
        from mujoco.rendering.filament.renderer import Renderer
        from mujoco.studio import native_viewer_cc, window  # noqa: F401

        self.env, self.text, self.repeat, self.W, self.H = env, text, repeat, width, height
        self.model = look_model(env.cfg)
        self.data, self._sim = mujoco.MjData(self.model), mujoco.MjData(env.model)
        cfg = mjrf.ContextConfig()
        cfg.graphics_api = mjrf.GraphicsApi.GRAPHICS_API_VULKAN
        self._ctx = mjrf.Context(cfg)
        self._renderer = Renderer(self._ctx)
        self._scene = native_viewer_cc.ModelScene(self._renderer.context, self._renderer.scene("scn"), self.model)
        self._buf = np.zeros((height, width, 3), np.uint8)
        self._renderer.target("out", self._buf)
        if isinstance(camera, str):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = self.model.camera(camera).id
            camera = cam
        self.cam, self.opt, self.pert = camera, mujoco.MjvOption(), mujoco.MjvPerturb()
        self.frames: list[np.ndarray] = []
        self.trail: list[np.ndarray] = []
        self.calls: list[np.ndarray] = []
        self._n = 0

    def __call__(self, env) -> None:
        from mujoco.rendering.filament.renderer import View

        ball = env.ball_state()[0, 0:3].copy()
        self.trail = (self.trail + [ball])[-40:]
        if env.n_plans[0] > self._n:
            self._n = int(env.n_plans[0])
            self.calls = (self.calls + [ball])[-25:]
        env.batch.get_data(0, self._sim)
        d = self.data
        d.qpos[:], d.qvel[:], d.mocap_pos[:] = self._sim.qpos, self._sim.qvel, self._sim.mocap_pos
        mujoco.mj_forward(self.model, d)
        g = [_sphere(p, 0.008, (1.0, 1.0, 1.0, 0.7)) for p in self.trail]
        if env.ekf.started[0] and not env.finished[0]:
            path = env.traj[env.nd::2, 0, 0:3]
            g += [_line(a, b, 0.004, (0.11, 0.69, 0.48, 1.0)) for a, b in zip(path[:-1], path[1:])
                  if a[2] > 0.5 and -0.2 < a[0] < env.L + 0.2]
        g += [_sphere(p, 0.012, (0.93, 0.63, 0.0, 1.0)) for p in self.calls]
        self._scene.sync(self.W, self.H, self.model, d, self.pert, self.cam, self.opt, g)
        self._renderer.render([View(scene="scn", target="out", camera=mujoco.mjv_camera2GLCamera(self.model, d, self.cam),
                                    headlight=True, headlight_intensity=0.5 * self._scene.get_headlight_intensity())])
        img = self._buf.copy()
        if self.text:
            img = label(img, self.text, size=22)
        img = label(img, f"t = {env.k * env.h:5.2f} s   returns {int(env.n_hits[0])}", size=18, xy=(10, img.shape[0] - 34))
        self.frames.extend([img] * self.repeat)

    def close(self) -> None:
        del self._scene, self._renderer, self._ctx
