"""M3: interception controller.  Catch rates, controller diagnostics, ablations, videos.

Controller: predictor -> IK for the hand pose at the predicted catch point -> joint-space
QP (OSQP when limits are active) -> discrete computed torque; replanning at a fixed 50 Hz
(the trigger study is M4).  Jaws close 50 ms before the predicted contact or on a
contact-sensor event.

Outputs: results/m3_catch.json, results/m3_catch.png, results/m3_<obj>.mp4.
"""
from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np

from card_catcher import RESULTS, config
from card_catcher.envs.catch import DEFAULT_PARAMS as PARAMS
from card_catcher.envs.catch import DEFAULT_PREDICTOR as PREDICTOR
from card_catcher.envs.catch import CatchEnv, CatchParams, run_fixed_rate
from card_catcher.envs.flight import generate
from card_catcher.estim.predictor import make_predictor
from card_catcher.eval import plotstyle as ps
from card_catcher.eval import render

SUCTION_FORCE = {"card": 1.0, "plate": 20.0}      # N per contact point (card weighs 0.018 N, plate 0.59 N)


def run(obj: str, kind: str, cfg: dict, n: int, seed: int, sim_override: dict | None = None, batch: int = 128,
        params: CatchParams | None = None) -> dict:
    c = dict(cfg, sim=dict(cfg["sim"], **(sim_override or {})))
    D = generate(obj, n, c, seed=400 + seed)
    outs, wall, steps = [], 0.0, 0
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        env = CatchEnv(obj, c, len(idx), make_predictor(kind, obj, len(idx), c),
                       params or PARAMS[obj], seed=seed, sim_override=None)
        t0 = time.time()
        outs.append(run_fixed_rate(env, D, idx))
        wall += time.time() - t0
        steps += env.k * len(idx)
    o = {k: np.concatenate([x[k] for x in outs]) for k in outs[0]}
    return {
        "caught": float(o["caught"].mean()), "clean": float(o["clean"].mean()),
        "final_pen_mm_caught": float(np.median(o["final_pen"][o["caught"]]) * 1e3) if o["caught"].any() else 0.0,
        "retained": float(o["retained"].mean()), "touched": float(o["touched"].mean()),
        "tcp_err_mm_median": float(np.nanmedian(o["tcp_err_at_cross"]) * 1e3), "tcp_err_mm_p90": float(np.nanquantile(o["tcp_err_at_cross"], 0.9) * 1e3),
        "pred_err_mm_median": float(np.nanmedian(o["pred_err_last"]) * 1e3), "plans_per_throw": float(o["n_plans"].mean()),
        "late_throws": float((o["late_plans"] > 0).mean()), "sat_steps_per_throw": float(o["sat_steps"].mean()),
        "ms_per_world_step": 1e3 * wall / steps, "n": int(n),
    }


def agg(rows: list[dict]) -> dict:
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}


def video(obj: str, kind: str, cfg: dict, seed: int, path, n_show: int = 3, params: CatchParams | None = None,
          key: str = "caught", tag: str = "") -> None:
    D = generate(obj, 8, cfg, seed=450 + seed)
    env = CatchEnv(obj, cfg, 8, make_predictor(kind, obj, 8, cfg), params or PARAMS[obj], seed=seed)
    r = mujoco.Renderer(env.model, 480, 640)
    data = mujoco.MjData(env.model)
    shots = {i: [] for i in range(8)}
    cams = [render.free_camera([0.85, 0.0, 0.85], 2.3, 90.0, -12.0), render.free_camera(env.plane.c0, 0.75, 140.0, -15.0)]

    def cb(e):
        for i in range(8):
            e.batch.get_data(i, data)
            mujoco.mj_forward(e.model, data)
            fr = []
            for cam in cams:
                r.update_scene(data, cam)
                fr.append(r.render().copy())
            shots[i].append(np.concatenate(fr, axis=1))

    out = run_fixed_rate(env, D, np.arange(8), frame_cb=cb)
    order = np.argsort(~out[key], kind="stable")[:n_show]               # caught throws first
    frames = []
    for i in order:
        txt = f"{obj}{tag}, {kind} predictor, 50 Hz replanning: " + ("CAUGHT" if out[key][i] else "missed")
        col = (120, 255, 120) if out[key][i] else (255, 90, 90)
        for f in shots[i]:
            frames += [render.label(f, txt, col)] * 2                 # 8 ms steps at 30 fps x 2 -> about 1/8 real time
        frames += [frames[-1]] * 15
    render.write_mp4(path, frames, 30)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="throws per seed")
    ap.add_argument("--n-ablation", type=int, default=128)
    ap.add_argument("--objects", nargs="*", default=["ball", "card", "plate"])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    cfg = config.load()
    out = {"n_per_seed": args.n, "seeds": cfg["seeds"], "main": {}, "ablation_predictor": {}, "ablation_contacts": {}, "stretch_suction": {}}
    path = RESULTS / "m3_catch.json"

    for obj in args.objects:
        kinds = ["oracle", "physics"] + (["hybrid"] if PREDICTOR[obj] == "hybrid" else [])
        for kind in kinds:
            rows = [run(obj, kind, cfg, args.n, s) for s in cfg["seeds"]]
            a = agg(rows)
            (out["main"] if kind == PREDICTOR[obj] else out["ablation_predictor"]).setdefault(obj, {})[kind] = a
            print(f"{obj:5s} {kind:8s}: caught {a['caught'][0]*100:5.1f} +- {a['caught'][1]*100:.1f} %  clean {a['clean'][0]*100:5.1f} %  retained {a['retained'][0]*100:5.1f} %  "
                  f"TCP err {a['tcp_err_mm_median'][0]:.1f} mm (p90 {a['tcp_err_mm_p90'][0]:.1f})  plans/throw {a['plans_per_throw'][0]:.1f}  "
                  f"late {a['late_throws'][0]*100:.0f} %  sat/throw {a['sat_steps_per_throw'][0]:.2f}  {a['ms_per_world_step'][0]:.2f} ms/step", flush=True)
            path.write_text(json.dumps(out, indent=2))

    if not args.skip_ablation:
        # speculative x timestep, oracle predictor (isolates contact physics from estimation)
        for obj in [o for o in ("ball", "card") if o in args.objects]:
            for dt in (0.008, 0.016):
                for spec in (True, False):
                    rows = [run(obj, "oracle", cfg, args.n_ablation, s, {"timestep": dt, "speculative": spec}) for s in cfg["seeds"]]
                    a = agg(rows)
                    out["ablation_contacts"].setdefault(obj, {})[f"{dt*1e3:.0f}ms_{'on' if spec else 'off'}"] = a
                    print(f"contacts {obj:5s} dt={dt*1e3:.0f} ms speculative={'on ' if spec else 'off'}: caught {a['caught'][0]*100:5.1f} +- "
                          f"{a['caught'][1]*100:.1f} %  clean {a['clean'][0]*100:5.1f} +- {a['clean'][1]*100:.1f} %  pen {a['final_pen_mm_caught'][0]:.2f} mm  "
                          f"retained {a['retained'][0]*100:5.1f} %  {a['ms_per_world_step'][0]:.2f} ms/step", flush=True)
                    path.write_text(json.dumps(out, indent=2))
        # stretch: suction disc (4.0 geom adhesion) instead of the parallel jaws; metric = retained
        import dataclasses

        for obj in [o for o in ("card", "plate") if o in args.objects]:
            prm = dataclasses.replace(PARAMS[obj], effector="suction", suction_force=SUCTION_FORCE[obj])
            for kind in ("oracle", PREDICTOR[obj]):
                a = agg([run(obj, kind, cfg, args.n_ablation, s, params=prm) for s in cfg["seeds"]])
                out["stretch_suction"].setdefault(obj, {})[kind] = a
                print(f"suction  {obj:5s} {kind:8s}: retained {a['retained'][0]*100:5.1f} +- {a['retained'][1]*100:.1f} %", flush=True)
                path.write_text(json.dumps(out, indent=2))
    plot(out)
    if not args.no_video:
        for obj in [o for o in ("ball", "card") if o in args.objects]:
            video(obj, PREDICTOR[obj], cfg, 0, RESULTS / f"m3_{obj}.mp4")
            print("video:", RESULTS / f"m3_{obj}.mp4")


def plot(out: dict) -> None:
    ps.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    a = ax[0]
    names = {"oracle": "oracle (true catch point)", "physics": "physics only (EKF)", "hybrid": "hybrid (physics + residual)"}
    cols = {"oracle": ps.MUTED, "physics": ps.METHOD["physics"], "hybrid": ps.METHOD["hybrid"]}
    objs = [o for o in ("ball", "card", "plate") if o in out["main"]]
    w = 0.26
    for j, kind in enumerate(("oracle", "physics", "hybrid")):
        xs, ys, es = [], [], []
        for i, obj in enumerate(objs):
            r = {**out["main"].get(obj, {}), **out["ablation_predictor"].get(obj, {})}.get(kind)
            if r is None:
                continue
            xs.append(i + (j - 1) * w)
            ys.append(r["caught"][0] * 100)
            es.append(r["caught"][1] * 100)
        a.bar(xs, ys, w * 0.92, yerr=es, color=cols[kind], label=names[kind], error_kw={"ecolor": ps.INK2, "lw": 1})
        for x, y in zip(xs, ys):
            a.text(x, y + 2.5, f"{y:.0f}", ha="center", fontsize=8.5, color=ps.INK2)
    a.axhline(90, color=ps.INK2, lw=1, ls="--")
    a.set_xticks(range(len(objs)), objs)
    a.set(ylabel="catch rate [%]", ylim=(0, 108), title="Catch rate by object and predictor (50 Hz replanning)")
    a.legend(loc="center right")
    a.grid(axis="x", visible=False)

    a = ax[1]
    ab = out.get("ablation_contacts", {})
    labels = ["8 ms\non", "8 ms\noff", "16 ms\non", "16 ms\noff"]
    keys = ["8ms_on", "8ms_off", "16ms_on", "16ms_off"]
    for j, obj in enumerate(("ball", "card")):
        if obj not in ab:
            continue
        ys = [ab[obj][k]["clean"][0] * 100 for k in keys]
        es = [ab[obj][k]["clean"][1] * 100 for k in keys]
        xs = np.arange(4) + (j - 0.5) * 0.36
        a.bar(xs, ys, 0.33, yerr=es, color=ps.OBJECT[obj], label=obj, error_kw={"ecolor": ps.INK2, "lw": 1})
        for x, y in zip(xs, ys):
            a.text(x, y + 2.5, f"{y:.0f}", ha="center", fontsize=8.5, color=ps.INK2)
    a.set_xticks(range(4), labels)
    a.set(ylabel="caught without interpenetration [%]", ylim=(0, 108), xlabel="timestep / speculative contacts",
          title="Ablation: speculative contacts x timestep (oracle predictor)")
    a.legend(loc="upper right")
    a.grid(axis="x", visible=False)
    fig.tight_layout()
    ps.save(fig, RESULTS / "m3_catch.png")
    print("figure:", RESULTS / "m3_catch.png")


if __name__ == "__main__":
    main()
