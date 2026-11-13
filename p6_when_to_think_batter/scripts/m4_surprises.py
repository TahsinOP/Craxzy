"""M4: surprises.  A spin change or a gust in mid-flight; which triggers react?

  1. train    : a second trigger policy on pitches of which half carry a surprise
                (the M3 policy never saw one; both are evaluated);
  2. pareto   : hit quality against planner calls on surprised pitches, per trigger;
  3. reaction : every trigger at the setting that costs about BUDGET planner calls on ordinary
                pitches (from M3), on the same pitches with and without the surprise:
                  - quality lost to the surprise,
                  - extra planner calls spent because of it,
                  - latency from the onset of the surprise to the next planner call, against
                    the same statistic on the same pitches *without* the surprise (placebo):
                    a trigger that reacts calls the planner sooner than it would have anyway;
  4. video    : gusts under a fixed low rate, the innovation trigger and the learned trigger.

Outputs: data/models/trigger_surprise.pt, results/m4.json, m4_pareto.png, m4_reaction.png, m4_surprise.mp4.

    env -u PYTHONPATH .venv/bin/python scripts/m4_surprises.py [--n 200] [--ppo-iters 150]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from batter import RESULTS, config
from batter.envs.cage import BatEnv, EnvParams, run_episode
from batter.estim.flight import FlightModel
from batter.eval import plotstyle as ps
from batter.eval import render
from batter.eval.runner import evaluate, make_trigger, mean_std, summarize
from batter.eval.sweeps import KINDS, LABEL, LAMS, at_budget, policy_path, sweep, train_ppo
from batter.models.pitch import generate
from batter.models.scene import build_cage_model

BUDGET = 4.0
WINDOW = 0.15          # s: "reacted" = called the planner within this long after the onset
SURPRISES = ["spin", "gust"]
ORDER = ["periodic", "variance", "drift", "innovation", "learned", "learned_s"]
NAMES = {**LABEL, "learned": "learned (PPO, never saw a surprise)", "learned_s": "learned (PPO, trained with surprises)"}
COLORS = {**ps.TRIGGER, "learned_s": ps.AQUA, "learned": "#7fd6b5"}


def latency(out: dict, h: float) -> np.ndarray:
    """Time from the surprise onset to the first planner call after it [s]; inf if there is none."""
    lat = np.full(len(out["hit"]), np.inf)
    for i, steps in enumerate(out["plan_steps"]):
        later = np.asarray(steps)[np.asarray(steps) > out["s_step"][i]]
        if later.size:
            lat[i] = (later[0] - out["s_step"][i]) * h
    return lat


def reaction(cfg: dict, fm, specs: dict, n: int, kind: str) -> dict:
    res = {}
    for name, spec in specs.items():
        per, lat_s, lat_p = [], [], []
        for seed in cfg["seeds"]:
            Ps = generate(fm, cfg, n, 800 + seed, surprise=kind)
            Pn = generate(fm, cfg, n, 800 + seed, surprise="none")           # the same pitches, no surprise
            os_, on = evaluate(cfg, spec, Ps, seed), evaluate(cfg, spec, Pn, seed)
            on["s_step"] = os_["s_step"]                                      # placebo onset
            ls, lp = latency(os_, fm.h), latency(on, fm.h)
            lat_s.append(ls), lat_p.append(lp)
            ss, sn = summarize(os_), summarize(on)
            per.append({"quality_surprise": ss["quality"], "quality_plain": sn["quality"], "hit_surprise": ss["hit"],
                        "hit_plain": sn["hit"], "on_target_surprise": ss["on_target"], "on_target_plain": sn["on_target"],
                        "calls_surprise": ss["plans"], "calls_plain": sn["plans"], "extra_calls": ss["plans"] - sn["plans"],
                        "react": float((ls <= WINDOW).mean()), "react_placebo": float((lp <= WINDOW).mean()),
                        "latency_med_ms": float(1e3 * np.median(ls)), "latency_placebo_med_ms": float(1e3 * np.median(lp)),
                        "no_call_after": float(np.isinf(ls).mean())})
        row = mean_std(per)
        row["latency_s"] = np.sort(np.concatenate(lat_s)).tolist()
        row["latency_p"] = np.sort(np.concatenate(lat_p)).tolist()
        res[name] = row
        fmt = lambda v: "none" if not np.isfinite(v) else f"{v:.0f} ms"
        print(f"  [{kind}] {NAMES[name]:38s}: quality {row['quality_plain'][0]:.3f} -> {row['quality_surprise'][0]:.3f}   calls "
              f"{row['calls_plain'][0]:.1f} -> {row['calls_surprise'][0]:.1f}   called within {1e3 * WINDOW:.0f} ms: "
              f"{100 * row['react'][0]:.0f} % (placebo {100 * row['react_placebo'][0]:.0f} %)   median latency "
              f"{fmt(row['latency_med_ms'][0])} (placebo {fmt(row['latency_placebo_med_ms'][0])})", flush=True)
    return res


def pareto_plot(res: dict) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    fig, ax = plt.subplots(1, len(SURPRISES), figsize=(6.2 * len(SURPRISES), 4.2), sharey=True)
    markers = {"periodic": "o", "innovation": "s", "variance": "^", "drift": "v", "learned": "D", "learned_s": "D"}
    for a, kind in zip(ax, SURPRISES):
        for name in ORDER:
            rows = sorted(res["pareto"][kind][name], key=lambda r: r["plans"][0])
            x = [r["plans"][0] for r in rows]
            y = np.array([r["quality"] for r in rows])
            a.errorbar(x, y[:, 0], y[:, 1], color=COLORS[name], marker=markers[name], label=NAMES[name], capsize=2,
                       ls="--" if name == "learned" else "-", lw=2.4 if name in ("learned_s", "periodic") else 1.5)
        a.set(xscale="log", xlabel="planner calls per pitch", title=f"Pitches with a {'spin change' if kind == 'spin' else 'gust'} in mid-flight")
    ax[0].set_ylabel("hit quality")
    ax[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    ps.save(fig, RESULTS / "m4_pareto.png")


def reaction_plot(res: dict) -> None:
    import matplotlib.pyplot as plt

    ps.apply()
    fig, ax = plt.subplots(2, len(SURPRISES), figsize=(6.4 * len(SURPRISES), 7.4))
    for c, kind in enumerate(SURPRISES):
        r = res["reaction"][kind]
        a = ax[0, c]
        grid = np.linspace(0.0, 0.4, 81)
        for name in ORDER:
            ls_, lp_ = np.array(r[name]["latency_s"]), np.array(r[name]["latency_p"])
            a.plot(1e3 * grid, [100 * np.mean(ls_ <= g) for g in grid], color=COLORS[name], label=NAMES[name],
                   lw=2.4 if name in ("learned_s", "innovation") else 1.5)
            a.plot(1e3 * grid, [100 * np.mean(lp_ <= g) for g in grid], color=COLORS[name], ls=":", lw=1.0)
        a.plot([], [], color=ps.INK2, ls=":", lw=1.0, label="same pitches, no surprise (placebo)")
        a.set(xlabel="time since the onset of the surprise [ms]", ylabel="pitches with a planner call by then [%]",
              title=f"{'Spin change' if kind == 'spin' else 'Gust'}: latency to the next planner call")
        if c == 0:
            a.legend(fontsize=7)
        a = ax[1, c]
        x = np.arange(len(ORDER))
        plain = np.array([r[n]["quality_plain"] for n in ORDER])
        sur = np.array([r[n]["quality_surprise"] for n in ORDER])
        a.bar(x - 0.2, plain[:, 0], 0.38, yerr=plain[:, 1], color=ps.GRID, edgecolor=ps.AXIS, label="same pitches, no surprise", capsize=2)
        a.bar(x + 0.2, sur[:, 0], 0.38, yerr=sur[:, 1], color=[COLORS[n] for n in ORDER], label="with the surprise", capsize=2)
        for xi, n in zip(x, ORDER):
            a.text(xi, 1.02, f"{r[n]['calls_plain'][0]:.1f}→{r[n]['calls_surprise'][0]:.1f}", ha="center", fontsize=7.5, color=ps.INK2)
        a.set_xticks(x, ["fixed\nrate", "variance", "drift", "innovation", "learned", "learned\n(+surprises)"], fontsize=8)
        a.set(ylabel="hit quality", ylim=(0, 1.1), title="Quality at about 4 calls per pitch (numbers: calls without → with)")
        if c == 0:
            a.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    ps.save(fig, RESULTS / "m4_reaction.png")


VIDEO = ["periodic", "innovation", "learned_s"]


def video(cfg: dict, specs: dict, renderer: str = "filament") -> None:
    """The same four gust pitches under a fixed rate, the innovation trigger and the learned trigger, side by side."""
    cam = render.free_camera([2.9, 0.0, 0.8], 7.2, 100.0, -14.0)
    frames = {}
    for name in VIDEO:
        env = BatEnv(cfg, 1, EnvParams(post_steps=60, max_time=1.5), seed=0)
        rec = render.make_recorder(renderer, env, cam, 640, 420, text=f"{NAMES[name]}", repeat=3)
        P = generate(env.fm, cfg, 4, seed=21, surprise="gust")
        for i in range(4):
            run_episode(env, P.take(np.array([i])), make_trigger(specs[name], 0), frame_cb=rec)
        frames[name] = rec.pitches
        rec.close()
    out = render.side_by_side([frames[k] for k in VIDEO])
    render.write_mp4(RESULTS / "m4_surprise.mp4", out, fps=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ppo-iters", type=int, default=150)
    ap.add_argument("--ppo-batch", type=int, default=192)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--renderer", default="filament", choices=["filament", "classic"], help="renderer for the video")
    ap.add_argument("--video-only", action="store_true", help="only render the video (needs results/m4.json)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = config.load(overrides=args.set)
    fm = FlightModel.identify(build_cage_model(cfg))
    m3 = json.loads((RESULTS / "m3.json").read_text())        # ordinary pitches: needed to match the budgets
    path = RESULTS / "m4.json"
    if args.video_only:
        sp = json.loads(path.read_text())["specs"]
        which = {"learned": "nominal", "learned_s": "surprise"}
        video(cfg, renderer=args.renderer, specs={k: ((v[0], v[1]) if k not in which else ("learned", (str(policy_path(which[k])), v[1])))
                    for k, v in sp.items() if k in VIDEO})
        return
    res: dict = {"n": args.n, "pareto": {}, "reaction": {}, "budget": BUDGET, "window_s": WINDOW}

    train_ppo("nominal", cfg, fm, args.ppo_iters, args.ppo_batch, surprise_frac=0.0)
    train_ppo("surprise", cfg, fm, args.ppo_iters, args.ppo_batch, surprise_frac=0.5, seed=1, force=args.retrain)
    learned = lambda which: (lambda lam: ("learned", (str(policy_path(which)), lam)))

    for kind in SURPRISES:
        out: dict = {}
        res["pareto"][kind] = out
        dump = lambda: path.write_text(json.dumps(res, indent=2))
        for k, settings in KINDS.items():
            sweep(cfg, fm, k, settings, args.n, kind, out, path, tag=f"[{kind}] ")
        sweep(cfg, fm, "learned", LAMS, args.n, kind, out, path, make_spec=learned("nominal"), tag=f"[{kind}] ")
        rows = sweep(cfg, fm, "learned_s", LAMS, args.n, kind, {}, RESULTS / "m4_tmp.json", make_spec=learned("surprise"), tag=f"[{kind}] ")
        out["learned_s"] = rows
        dump()
    (RESULTS / "m4_tmp.json").unlink(missing_ok=True)

    # the surprise-trained policy on ordinary pitches, to find its setting at the budget
    plain_s = sweep(cfg, fm, "learned_s", LAMS, args.n, "none", {}, RESULTS / "m4_tmp.json", make_spec=learned("surprise"), tag="[none] ")
    (RESULTS / "m4_tmp.json").unlink(missing_ok=True)
    res["learned_s_plain"] = plain_s
    specs = {k: (k, at_budget(m3[k], BUDGET)["setting"]) for k in KINDS}
    specs["learned"] = learned("nominal")(at_budget(m3["learned"], BUDGET)["setting"])
    specs["learned_s"] = learned("surprise")(at_budget(plain_s, BUDGET)["setting"])
    res["specs"] = {k: [v[0], v[1] if not isinstance(v[1], tuple) else v[1][1]] for k, v in specs.items()}
    for kind in SURPRISES:
        res["reaction"][kind] = reaction(cfg, fm, specs, args.n, kind)
        path.write_text(json.dumps(res, indent=2))
    pareto_plot(res)
    reaction_plot(res)
    if not args.no_video:
        video(cfg, specs, renderer=args.renderer)


if __name__ == "__main__":
    main()
