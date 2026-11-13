"""M0: speculative contacts vs tunnelling of the 0.7 mm card through the gripper.

The arm holds the ready pose with the jaws open.  A card is thrown face-first
along the jaw axis at the outside of one fingertip, so that the only solid in
its path is that finger (9-11 mm of finger plus the 8 mm rubber pad behind it;
the card's edges clear the palm).  Per step the card travels v*dt = 1.6-4 cm at
8 ms and 3.2-8 cm at 16 ms: far more than its own thickness (0.7 mm) and more
than the finger's.

Tunnelling criterion (rigorous: a rigid card contains its own centre of mass):
the card's COM crosses the mid-plane of the fingertip *pad* -- which lies behind
the whole thickness of the finger -- at a point at least 2 mm inside the pad's
box laterally and 4 mm inside it along the flight direction.  Reaching that
point from outside means the COM has gone through >= 13 mm of solid; no amount
of soft-contact penetration (observed: 1-3 mm) can be mistaken for it.

Outputs: results/m0_tunnel.json, results/m0_tunnel.mp4 (2x2: flag off/on x 8/16 ms).
"""
from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from card_catcher import RESULTS, config
from card_catcher.control.arm import Arm, ready_configuration
from card_catcher.eval import render
from card_catcher.models.objects import OBJECTS
from card_catcher.models.scene import PARK_POS, build_catch_model
from card_catcher.models.throws import _quat_from_mat, solve_launch_velocity

STANDOFF = 0.12          # launch distance from the finger's outer face [m]
Y_OUTER = 0.0585         # outer face of the finger near its tip, hand frame [m]
AIM_X = (-0.003, 0.003)  # aim region on the fingertip, hand frame [m]
AIM_Z = (0.100, 0.105)   # the card (half-width 31.5 mm along hand z) then clears the palm (z <= 0.066)


class Scenario:
    def __init__(self, cfg, dt: float, speculative: bool):
        sim = dict(cfg["sim"], timestep=dt, speculative=speculative)
        self.m = build_catch_model("card", sim)
        self.arm = Arm(self.m)
        self.ix = self.arm.ix
        self.d = mujoco.MjData(self.m)
        self.q_ready = ready_configuration(self.arm, config.Plane.from_cfg(cfg))
        self.reset()
        self.Rh = self.d.xmat[self.ix.hand_body].reshape(3, 3).copy()
        self.ph = self.d.xpos[self.ix.hand_body].copy()

    def reset(self):
        m, d, ix = self.m, self.d, self.ix
        mujoco.mj_resetData(m, d)
        d.qpos[ix.arm_qadr] = self.q_ready
        d.qpos[ix.finger_qadr] = 0.04
        d.ctrl[ix.grip_act] = 255
        d.qpos[ix.obj_qadr:ix.obj_qadr + 3] = PARK_POS
        mujoco.mj_forward(m, d)

    def pad_box(self) -> tuple[np.ndarray, np.ndarray]:
        """(min, max) corners of the left fingertip pad (largest box geom), in the hand frame."""
        m, d = self.m, self.d
        bid = m.body("left_finger").id
        boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        g = max(boxes, key=lambda g: np.prod(m.geom_size[g]))
        sz = m.geom_size[g]
        V = np.array([[a * sz[0], b * sz[1], c * sz[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        H = (d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T - self.ph) @ self.Rh
        return H.min(0), H.max(0)

    def hand_to_world(self, x):
        return self.ph + self.Rh @ np.asarray(x, float)

    def throw(self, aim_xz, speed: float, nstep_after: int = 12, frames_cb=None):
        """Throw the card face-first along -y_hand at the fingertip point (x, Y_OUTER, z).
        Returns the card COM history in the hand frame (K, 3) and the name of the
        first robot body the card touches (None if it touches nothing)."""
        m, d, ix = self.m, self.d, self.ix
        self.reset()
        dt = m.opt.timestep
        target = self.hand_to_world([aim_xz[0], Y_OUTER, aim_xz[1]])
        start = self.hand_to_world([aim_xz[0], Y_OUTER + STANDOFF, aim_xz[1]])
        T = STANDOFF / speed
        kd_face = OBJECTS["card"].kd_axis(np.array([0.0, 0.0, 1.0]), cfg_density)
        v0 = solve_launch_velocity(start[None], target[None], np.array([T]), kd_face, dt)[0]
        # card normal (body z) along the launch velocity: face-first
        z = v0 / np.linalg.norm(v0)
        x = np.cross(self.Rh[:, 0], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], axis=1)
        d.qpos[ix.obj_qadr:ix.obj_qadr + 3] = start
        d.qpos[ix.obj_qadr + 3:ix.obj_qadr + 7] = _quat_from_mat(R[None])[0]
        d.qvel[ix.obj_dadr:ix.obj_dadr + 3] = v0
        d.qvel[ix.obj_dadr + 3:ix.obj_dadr + 6] = 0.0
        hist = []
        first = None
        nstep = int(np.ceil(T / dt)) + nstep_after
        zero = np.zeros(7)
        for _ in range(nstep):
            mujoco.mj_step1(m, d)
            hist.append((d.qpos[ix.obj_qadr:ix.obj_qadr + 3] - self.ph) @ self.Rh)
            if first is None:
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    if ix.obj_body in (b1, b2) and c.dist < 0:
                        first = m.body(b2 if b1 == ix.obj_body else b1).name
                        break
            if frames_cb is not None:
                frames_cb(self)
            tau, _ = self.arm.torque(d.M, d.qfrc_bias, d.qfrc_passive, d.qpos[ix.arm_qadr], d.qvel[ix.arm_dadr],
                                     self.q_ready, zero, zero)
            d.ctrl[:7] = tau
            mujoco.mj_step2(m, d)
        hist.append((d.qpos[ix.obj_qadr:ix.obj_qadr + 3] - self.ph) @ self.Rh)
        return np.array(hist), first


def tunnelled(hist: np.ndarray, pad: tuple[np.ndarray, np.ndarray]) -> bool:
    """COM crosses the pad's mid-plane at a point >= 2 mm inside the pad box laterally."""
    lo, hi = pad
    y_mid = 0.5 * (lo[1] + hi[1])
    y = hist[:, 1]
    for k in np.flatnonzero((y[:-1] > y_mid) & (y[1:] <= y_mid)):
        a = (y[k] - y_mid) / (y[k] - y[k + 1])
        pt = hist[k] + a * (hist[k + 1] - hist[k])
        if lo[0] + 0.002 <= pt[0] <= hi[0] - 0.002 and lo[2] + 0.002 <= pt[2] <= hi[2] - 0.002:
            return True
    return False


def main():
    global cfg_density
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="throws per condition and seed")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    cfg = config.load()
    cfg_density = cfg["sim"]["density"]
    conds = [(dt, spec) for dt in (0.008, 0.016) for spec in (False, True)]
    scen = {c: Scenario(cfg, *c) for c in conds}
    pad = scen[conds[0]].pad_box()
    assert pad[1][1] - pad[0][1] >= 0.0079, "pad must be 8 mm thick along the flight direction"
    assert pad[0][0] + 0.002 < AIM_X[0] and AIM_X[1] < pad[1][0] - 0.002 and pad[0][2] + 0.002 < AIM_Z[0] and AIM_Z[1] < pad[1][2] - 0.002

    out = {"n_per_seed": args.n, "seeds": cfg["seeds"], "conditions": []}
    for c in conds:
        rates, by_speed, first_body = [], {"2-3": [], "3-4": [], "4-5": []}, {}
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(seed)       # same throws for every condition
            hits = []
            for _ in range(args.n):
                aim = (rng.uniform(*AIM_X), rng.uniform(*AIM_Z))
                speed = rng.uniform(2.0, 5.0)
                hist, first = scen[c].throw(aim, speed)
                t = tunnelled(hist, pad)
                hits.append(t)
                first_body[first] = first_body.get(first, 0) + 1
                by_speed["2-3" if speed < 3 else "3-4" if speed < 4 else "4-5"].append(t)
            rates.append(float(np.mean(hits)))
        row = {"timestep_ms": c[0] * 1e3, "speculative": c[1], "tunnel_rate_mean": float(np.mean(rates)),
               "tunnel_rate_std": float(np.std(rates)), "per_seed": rates,
               "by_speed": {k: float(np.mean(v)) for k, v in by_speed.items()},
               "first_contact_body": {str(k): v for k, v in first_body.items()}}
        out["conditions"].append(row)
        print(f"dt={c[0]*1e3:4.0f} ms speculative={str(c[1]):5s}: tunnelling {row['tunnel_rate_mean']*100:5.1f} "
              f"+- {row['tunnel_rate_std']*100:.1f} %  by speed "
              + ", ".join(f"{k} m/s: {v*100:.0f}%" for k, v in row["by_speed"].items()) + f"  first contact: {row['first_contact_body']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m0_tunnel.json").write_text(json.dumps(out, indent=2))

    if not args.no_video:
        # one throw at 3 m/s, all four conditions on a common slow-motion timeline (1/16 real time)
        speed, aim = 3.0, (0.0, 0.1025)
        s0 = scen[conds[0]]
        cam = render.free_camera(s0.hand_to_world([0, 0.05, 0.085]), 0.42, 90.0, -8.0)
        tracks = {}
        for c in conds:
            s = scen[c]
            r = mujoco.Renderer(s.m, 480, 640)
            imgs = []

            def cb(sc, r=r, imgs=imgs):
                r.update_scene(sc.d, cam)
                imgs.append(r.render().copy())

            hist, _ = s.throw(aim, speed, nstep_after=int(0.12 / c[0]), frames_cb=cb)
            tracks[c] = (imgs, tunnelled(hist, pad))
            r.close()
        fps, slow = 30, 16.0
        t_end = min(len(tracks[c][0]) * c[0] for c in conds)
        frames = []
        for i in range(int(t_end * slow * fps)):
            t = i / (fps * slow)
            panels = []
            for c in conds:
                imgs, tun = tracks[c]
                img = imgs[min(int(t / c[0]), len(imgs) - 1)]
                txt = f"dt = {c[0]*1e3:.0f} ms, speculative {'ON' if c[1] else 'OFF'}: " + ("TUNNELS" if tun else "blocked")
                panels.append(render.label(img, txt, (255, 90, 90) if tun else (120, 255, 120)))
            frames.append(render.grid(panels, 2))
        frames += [frames[-1]] * fps
        path = render.write_mp4(RESULTS / "m0_tunnel.mp4", frames, fps)
        print("video:", path, {f"{c[0]*1e3:.0f}ms/{'on' if c[1] else 'off'}": tracks[c][1] for c in conds})


if __name__ == "__main__":
    main()
