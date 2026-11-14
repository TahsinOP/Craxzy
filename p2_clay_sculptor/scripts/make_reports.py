"""Write results/M0.md ... M4.md from the JSON results (numbers are never typed by hand).

    env -u PYTHONPATH .venv/bin/python scripts/make_reports.py
"""
import json

import numpy as np
from scipy import stats

from clay_sculptor import RESULTS


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pm(v, d=2, scale=1.0):
    return f"{scale * v[0]:.{d}f} ± {scale * v[1]:.{d}f}"


def paired(a, b) -> tuple[float, float, float, int]:
    """Mean of a - b over paired episodes (same target, same planner seed), its 95 %
    t-interval half-width, the two-sided p-value of the paired t-test, and n."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1) * se)
    p = float(stats.ttest_rel(a, b).pvalue) if se > 0 else 1.0
    return float(d.mean()), half, p, n


def write(name, lines):
    (RESULTS / name).write_text("\n".join(lines) + "\n")
    print("wrote", f"results/{name}")


LABEL = dict(oracle="oracle (MuJoCo in the loop)", prior="prior only (classical)", gnn="GNN (pure learning)",
             structured="structured: prior + GNN residual (hybrid)", unet="height-map U-Net (pure learning)", no_change="no change")

# ---- M0 ------------------------------------------------------------------------------------
m0 = load("m0_bench")
if m0:
    c, s, pr = m0["config"]["clay"], m0["config"]["sim"], m0["permanence"]
    L = ["# M0: clay bench", "",
         "Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m0_bench.py`", "",
         "## Scene",
         f"- Clay: {1e3 * c['size'][0]:.0f} x {1e3 * c['size'][1]:.0f} x {1e3 * c['size'][2]:.0f} mm block, 3D flex with a trilinear cage of "
         f"{c['cells'][0]} x {c['cells'][1]} x {c['cells'][2]} cells (bottom node layer pinned), {c['count'][0]} x {c['count'][1]} x {c['count'][2]} "
         f"interpolated vertices; Young's modulus {c['young'] / 1e3:.0f} kPa, Poisson ratio {c['poisson']}, yield strain {c['yield_strain']}, "
         f"density {c['density']:.0f} kg/m^3.",
         f"- Integrator `discrete`, timestep {1e3 * s['timestep']:.0f} ms, CG solver. {m0['speed']['nv']} degrees of freedom, "
         f"{pr['nplastic']} plastic state variables (6 per cage cell).",
         "- Menagerie Panda (no hand) with a tool attached through `MjSpec.attach`: flat press, roller, cross stamp. "
         "Gravity compensation on, `pid` servos with position and velocity setpoints.", "",
         "## Acceptance", "",
         "### 1. Scripted press, roll and stamp sequences produce repeatable shapes", "",
         "| sequence | actions | deepest point [mm] | highest rise [mm] | side bulge [mm] | cage volume ratio | second run bit-identical | "
         "height change for a 0.1 mm change of every action [mm] |", "|---|---|---|---|---|---|---|---|"]
    for k, v in m0["sequences"].items():
        L.append(f"| {k} | {len(v['actions'])} | {v['deepest_mm']:.2f} | {v['highest_rise_mm']:.2f} | {v['side_bulge_mm']:.2f} | "
                 f"{v['volume_ratio']:.4f} | {'yes' if v['repeat_bit_identical'] else 'NO'} | {v['sensitivity_mm_per_0p1mm']:.3f} |")
    L += ["", "The second run is in a separate environment on one thread; the first is on all threads. "
          "The last column shows the map from actions to shape is smooth: a 0.1 mm change of the commands changes the shape by a comparable amount.", "",
          "![height maps](m0_shapes.png)", "", "![renders](m0_render.png)", "",
          "### 2. Dents stay after the tool leaves", "",
          f"One press, commanded depth {1e3 * pr['press'][3]:.0f} mm, then {pr['hold_s']:.0f} s with the tool away.", "",
          "| clay | depth under the tool at maximum [mm] | when the primitive ends [mm] | after the hold [mm] |", "|---|---|---|---|",
          f"| plastic (yield strain {c['yield_strain']}) | {pr['plastic']['max_depth_mm']:.2f} | {pr['plastic']['after_primitive_mm']:.2f} | {pr['plastic']['after_hold_mm']:.2f} |",
          f"| elastic (yield off) | {pr['elastic']['max_depth_mm']:.2f} | {pr['elastic']['after_primitive_mm']:.2f} | {pr['elastic']['after_hold_mm']:.2f} |", "",
          f"The plastic dent moves by {pr['plastic']['drift_during_hold_mm']:.4f} mm during the hold. "
          f"Spring-back (maximum depth minus final depth) is {pr['plastic']['max_depth_mm'] - pr['plastic']['after_hold_mm']:.2f} mm, "
          f"{100 * (1 - pr['plastic']['after_hold_mm'] / pr['plastic']['max_depth_mm']):.0f} % of the stroke.", "",
          "![permanence](m0_permanence.png)", "",
          "### 3. State save/restore with `mjSTATE_PLASTIC` reproduces rollouts bit for bit", ""]
    r = m0["restore"]
    L += [f"The state is the `mjSTATE_INTEGRATION` vector ({r['full_state_size']} numbers: " +
          ", ".join(f"{k} {v}" for k, v in r["components"].items()) + ").", "",
          f"- restore the full state and replay: bit-identical = **{r['replay_bit_identical']}**",
          f"- `mujoco.batch` against plain `mj_step` on one `MjData`: bit-identical = **{r['batch_equals_single_thread']}**", "",
          "What an incomplete restore costs (same press replayed from a state with one component reset):", "",
          "| component left out | bit-identical | largest node position difference [mm] | largest height difference [mm] |", "|---|---|---|---|"]
    for k, v in r["omitted"].items():
        L.append(f"| {k} | {'yes' if v['bit_identical'] else 'no'} | {v['max_node_diff_mm']:.2e} | {v['max_height_diff_mm']:.2e} |")
    L += ["", "## Further checks", "",
          "### Rollouts skip the free-space moves", "",
          "Rollouts put the arm directly at the hover pose of each primitive. Executing the same sequences continuously, with the moves between primitives simulated:", "",
          "| sequence | largest height difference [mm] | RMS height difference [mm] |", "|---|---|---|"]
    for k, v in m0["continuous"].items():
        L.append(f"| {k} | {v['max_height_diff_mm']:.4f} | {v['rms_height_diff_mm']:.4f} |")
    L += ["", "### Tool tracking", "",
          "| servo | tool | commanded depth [mm] | depth reached [mm] | tip error at the end of the dwell (x, y, z) [mm] | largest error during the primitive [mm] | largest joint torque [N m] |",
          "|---|---|---|---|---|---|---|"]
    rows = [("pid", k, v) for k, v in m0["tracking"].items()] + [("PD only", k, v) for k, v in m0.get("tracking_pd_only", {}).items()]
    for servo, k, v in rows:
        L.append(f"| {servo} | {k} | {v['depth_commanded_mm']:.1f} | {v['depth_reached_mm']:.2f} | {', '.join(f'{x:+.2f}' for x in v['err_end_of_dwell_mm'])} | "
                 f"{', '.join(f'{x:.2f}' for x in v['err_max_mm'])} | {max(v['torque_max_nm']):.1f} |")
    sp = m0["speed"]
    L += ["", "### Speed", "",
          f"A press or stamp is {sp['steps']['press']} steps, a roll {sp['steps']['roller']}.", "",
          "| threads | press [s per rollout] | roll [s per rollout] |", "|---|---|---|"]
    for n in (1, 8, 16):
        L.append(f"| {n} | {sp[f'press_s_per_rollout_{n}thread']:.3f} | {sp[f'roller_s_per_rollout_{n}thread']:.3f} |")
    L += ["", "Video: `results/m0_bench.mp4` (the three sequences, executed continuously)."]
    dz = load("m0_discretisation")
    if dz:
        L += ["", "## Design check: which discretisation", "",
              "Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m0_discretisation.py`. One press with the flat press, then 1.5 s with the tool away.", "",
              "| discretisation | press [mm] | degrees of freedom | step time [ms] | share of the stroke that stays [%] | largest vertex motion after the primitive [mm] | "
              "smallest element volume / reference | inverted elements |", "|---|---|---|---|---|---|---|---|"]
        for r in dz:
            L.append(f"| {r['label']} | {r['depth_cmd_mm']:.0f} | {r['dof']} | {r['ms_per_step']:.1f} | {100 * r['retained']:.0f} | {r['creep_mm']:.3f} | "
                     f"{r['min_volume_ratio']:.2f} | {r['inverted']} |")
        L += ["", "The full-vertex flex creeps after unloading and inverts elements at 14 mm; the trilinear cage does neither. "
              "The share of the stroke that stays hardly depends on the yield strain, so the spring-back of the cage is a property of the discretisation."]
    write("M0.md", L)

# ---- M1 ------------------------------------------------------------------------------------
m1 = load("m1_oracle")
if m1:
    L = ["# M1: oracle MPC", "",
         "Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m1_oracle.py`", "",
         f"CEM over the five primitive parameters, population {m1['cem']['pop']}, {m1['cem']['iters']} iterations, {m1['cem']['elite']} elites, "
         f"{100 * m1['cem']['guided']:.0f} % of the first population from the guided proposal. Rollout model: MuJoCo itself. "
         "Cost: height-map RMSE to the target. Greedy receding horizon: plan one primitive, execute, observe, repeat; stop when no candidate improves.", "",
         f"Targets are made by hidden scripted demonstrations, so zero error is reachable. Seeds: {m1['seeds']} (mean ± std).", "",
         "| target | tool | demonstration | untouched block: RMSE [mm] | reached: RMSE [mm] | error left [%] | Chamfer [mm] | EMD [mm] | actions | oracle rollouts | planning time [s] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, t in m1["targets"].items():
        s = t["summary"]
        L.append(f"| {n} | {t['tool']} | {len(t['demo'])} action(s) | {t['untouched']['map_rmse_mm']:.2f} | {pm(s['map_rmse_mm'], 3)} | {pm(s['remaining'], 1, 100)} | "
                 f"{pm(s['chamfer_mm'], 3)} | {pm(s['emd_mm'], 3)} | {pm(s['n_actions'], 1)} | {pm(s['plan_calls'], 0)} | {pm(s['plan_time_s'], 0)} |")
    L += ["", "For scale, the untouched block against each target: " +
          "; ".join(f"{n}: Chamfer {t['untouched']['chamfer_mm']:.2f} mm, EMD {t['untouched']['emd_mm']:.2f} mm" for n, t in m1["targets"].items()) + ".", "",
          "![targets and reached shapes](m1_oracle.png)", "", "![error after each action](m1_oracle_curves.png)", "",
          "This is the upper bound for the learned-model planners of M3: same planner, same budget, exact model."]
    write("M1.md", L)

# ---- M2 ------------------------------------------------------------------------------------
m2, ds = load("m2_eval"), load("m2_dataset_main")
if m2:
    L = ["# M2: learned dynamics", "",
         "Reproduce: `scripts/m2_dataset.py`, `scripts/m2_train.py`, `scripts/m2_eval.py` (see `scripts/run_all.sh` for the arguments).", ""]
    if ds:
        L += [f"Data: {ds['episodes']} episodes of {ds['steps']} random primitives from the untouched block ({ds['transitions']} transitions, "
              f"{ds['failed']} failed), {ds['s_per_transition']:.2f} s per transition on CPU with `mujoco.batch`. "
              f"The last 200 episodes are held out: 100 for validation, 100 for the test below ({m2['n_test']} transitions).", ""]
    L += ["Models: (a) `gnn`, a graph network on the cage nodes; (b) `structured`, the swept-volume prior plus the same graph network on the residual; "
          "(c) `unet`, a U-Net on the 32 x 32 height map. Baselines: the prior alone, and predicting no change.", "",
          f"All errors are height-map RMSE in mm on held-out transitions, mean ± std over model seeds {m2['seeds']}.", "",
          "## One step and five steps", "",
          "| model | one step | press | roller | stamp | five steps, open loop | node position RMSE [mm] | parameters | time per prediction [ms] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in ("no_change", "prior", "gnn", "structured", "unet"):
        o = m2["one_step"][k]
        bt = o["by_tool"]
        f = lambda v: pm(v, 3) if isinstance(v, list) else f"{v:.3f}"
        five = m2["multi_step"][k][4]
        sp = m2["speed"].get(k, {})
        L.append(f"| {LABEL[k]} | {pm(o['map_rmse_mm'], 3) if k != 'no_change' else f'{o['map_rmse_mm'][0]:.3f}'} | {f(bt['press'])} | {f(bt['roller'])} | {f(bt['stamp'])} | "
                 f"{pm(five, 3) if k != 'no_change' else f'{five[0]:.3f}'} | {pm(o['node_rmse_mm'], 3) if o.get('node_rmse_mm') else '-'} | "
                 f"{sp.get('n_params', '-')} | {f'{sp['ms_per_prediction']:.2f}' if sp else '-'} |")
    if "oracle" in m2["speed"]:
        o = m2["speed"]["oracle"]
        L += ["", f"Oracle for comparison: {o['ms_per_prediction_1thread']:.0f} ms per press rollout on one thread, "
              f"{o['ms_per_prediction_16thread']:.0f} ms each when 32 run in parallel on all cores ({o['roller_ms_16thread']:.0f} ms for a roll). "
              "Model times are per candidate in a batch of 32 on 8 threads, height-map conversion included."]
    L += ["", "![errors](m2_errors.png)", "", "![examples](m2_examples.png)", "", "## Error against the horizon", "",
          "| primitives predicted open loop | " + " | ".join(str(i + 1) for i in range(len(m2["multi_step"]["no_change"]))) + " |",
          "|---|" + "---|" * len(m2["multi_step"]["no_change"])]
    for k in ("no_change", "prior", "gnn", "structured", "unet"):
        L.append(f"| {LABEL[k]} | " + " | ".join(f"{v[0]:.2f}" for v in m2["multi_step"][k]) + " |")
    if m2["training"]:
        L += ["", "## Training", "", "| model | seeds | epochs | training transitions | wall time per model [min] |", "|---|---|---|---|---|"]
        for k, runs in m2["training"].items():
            L.append(f"| {k} | {len(runs)} | {runs[0]['epochs']} | {runs[0]['n_train']} | {sum(r['wall_s'] for r in runs) / len(runs) / 60:.1f} |")
    if m2["shift"]:
        L += ["", "## Domain shift", "", "The same models, trained on the nominal clay, on clay with a different material (one-step height-map RMSE [mm]).", "",
              "| clay | transitions | no change | prior | GNN | structured | U-Net |", "|---|---|---|---|---|---|---|",
              f"| nominal | {m2['n_test']} | {m2['one_step']['no_change']['map_rmse_mm'][0]:.3f} | {m2['one_step']['prior']['map_rmse_mm'][0]:.3f} | "
              f"{pm(m2['one_step']['gnn']['map_rmse_mm'], 3)} | {pm(m2['one_step']['structured']['map_rmse_mm'], 3)} | {pm(m2['one_step']['unet']['map_rmse_mm'], 3)} |"]
        for n, v in m2["shift"].items():
            L.append(f"| {' '.join(v['overrides'])} | {v['n']} | {v['no_change'][0]:.3f} | {v['prior'][0]:.3f} | {pm(v['gnn'], 3)} | {pm(v['structured'], 3)} | {pm(v['unet'], 3)} |")
        L += ["", "![domain shift](m2_shift.png)"]
    write("M2.md", L)

# ---- M3 ------------------------------------------------------------------------------------
m3 = load("m3_mpc")
if m3:
    targets = [t for t in next(iter(m3["methods"].values())) if t != "overall"]
    L = ["# M3: learned-model MPC", "",
         "Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m3_mpc.py`", "",
         f"The planner of M1 (CEM, population {m3['cem']['pop']}, {m3['cem']['iters']} iterations, same guided proposal, same stopping rule) with each model "
         f"in place of the oracle. Every executed action is a real simulated primitive and the planner observes the true shape after it. "
         f"Planner seeds {m3["seeds"]}; planner seed i plans with the model trained with seed i mod 3. Mean ± std over planner seeds.", "",
         "## Shape error against planning time", "",
         "| rollout model | " + " | ".join(f"{t}: RMSE [mm]" for t in targets) + " | initial error left, all targets [%] | planning time per decision [s] | oracle rollouts for planning, all targets |",
         "|---|" + "---|" * (len(targets) + 3)]
    for k in ("oracle", "prior", "gnn", "structured", "unet"):
        if k in m3["methods"]:
            r = m3["methods"][k]
            L.append(f"| {LABEL[k]} | " + " | ".join(pm(r[t]["summary"]["map_rmse_mm"], 2) for t in targets) +
                     f" | {pm(r['overall']['remaining'], 1, 100)} | {pm(r['overall']['plan_time_per_decision_s'], 2)} | {r['overall']['plan_calls'][0]:.0f} |")
    if "untouched" in m3:
        L += ["", "Untouched block: " + ", ".join(f"{t} {m3['untouched'][t]['map_rmse_mm']:.2f} mm" for t in targets) + "."]
    rem = lambda k: [e["remaining"] for t in targets for e in sorted(m3["methods"][k][t]["episodes"], key=lambda e: e["seed"])]
    common = [k for k in ("oracle", "prior", "gnn", "unet") if k in m3["methods"] and len(rem(k)) == len(rem("structured"))]
    if common:
        L += ["", "## Paired comparison with the structured model", "",
              "Each episode of a method is paired with the structured model's episode on the same target with the same planner seed. "
              "Difference of the share of the initial error left (method minus structured; negative = the method is better), "
              "mean with 95 % t-interval, and the p-value of the paired t-test.", "",
              "| method | difference [percentage points] | p | pairs |", "|---|---|---|---|"]
        for k in common:
            d, h, pv, n = paired(rem(k), rem("structured"))
            L.append(f"| {LABEL[k]} | {100 * d:+.1f} ± {100 * h:.1f} | {pv:.3f} | {n} |")
    L += ["", "![trade-off](m3_mpc.png)", "", "![shapes](m3_mpc_shapes.png)", "", "## All metrics per target", ""]
    for t in targets:
        L += [f"### {t}", "", "| rollout model | height-map RMSE [mm] | Chamfer [mm] | EMD [mm] | actions | planning time per decision [s] |", "|---|---|---|---|---|---|"]
        for k in ("oracle", "prior", "gnn", "structured", "unet"):
            if k in m3["methods"]:
                s = m3["methods"][k][t]["summary"]
                L.append(f"| {LABEL[k]} | {pm(s['map_rmse_mm'], 3)} | {pm(s['chamfer_mm'], 3)} | {pm(s['emd_mm'], 3)} | {pm(s['n_actions'], 1)} | {pm(s['plan_time_per_decision_s'], 2)} |")
        L.append("")
    write("M3.md", L)

# ---- M4 ------------------------------------------------------------------------------------
m4 = load("m4_fallback")
if m4:
    cal = m4["calibration"]
    L = ["# M4: know when the model is wrong", "",
         "Reproduce: `env -u PYTHONPATH .venv/bin/python scripts/m4_fallback.py`", "",
         f"Deep ensemble: {len(m4['members'])} independently trained `{m4['kind']}` models. The planner searches with the ensemble mean. "
         f"The disagreement of a candidate is the RMS over pixels of the standard deviation of the members' predicted height maps; "
         f"the misranking probability is the largest, over the other top candidates, of the probability (normal fit to the members) that it is truly better than the chosen one. "
         f"When triggered, the {m4['top_m']} best candidates are re-ranked with oracle rollouts ({m4['top_m']} simulated primitives) and the truly best one is executed.", "",
         "## Is the disagreement informative?", "",
         f"On {cal['n']} held-out transitions the Spearman rank correlation between disagreement and the true error of the ensemble mean is "
         f"**{cal['spearman']:.2f}**. The quarter of transitions with the largest disagreement carries {100 * cal['top_quarter_share_of_error']:.0f} % of the total error. "
         f"Ensemble mean error {cal['ensemble_mean_rmse_mm']:.3f} mm against {cal['single_member_rmse_mm']:.3f} mm for a single member.", "",
         "| disagreement bin (mean) [mm] | " + " | ".join(f"{v:.3f}" for v in cal["spread_mm"]) + " |", "|---|" + "---|" * len(cal["spread_mm"]),
         "| true error (mean) [mm] | " + " | ".join(f"{v:.3f}" for v in cal["error_mm"]) + " |", "",
         "![calibration](m4_fallback_calibration.png)", "",
         "## Shape error against oracle calls", "",
         f"Three targets, seeds {m4['seeds']}. Error = share of the initial height-map RMSE that is left, averaged over targets (mean ± std over seeds).", "",
         "| planner | trigger fires at | initial error left [%] | oracle rollouts per episode | " +
         " | ".join(f"{t}: RMSE [mm]" for t in next(iter(m4["configs"].values()))["summary"]["by_target"]) + " |",
         "|---|---|---|---|" + "---|" * len(next(iter(m4["configs"].values()))["summary"]["by_target"])]
    for k, v in m4["configs"].items():
        s = v["summary"]
        L.append(f"| {k} | {100 * s['fire_rate']:.0f} % of decisions | {pm(s['remaining'], 1, 100)} | {pm(s['calls_per_episode'], 1)} | " +
                 " | ".join(pm(b["map_rmse_mm"], 2) for b in s["by_target"].values()) + " |")
    if "oracle" in m4:
        L.append(f"| oracle for every candidate (M1) | - | {pm(m4['oracle']['remaining'], 1, 100)} | {pm(m4['oracle']['calls_per_episode'], 1)} | " +
                 " | ".join("" for _ in next(iter(m4["configs"].values()))["summary"]["by_target"]) + " |")
    rem4 = lambda k: [e["remaining"] for t in m4["targets_order"] for e in sorted(m4["configs"][k]["episodes"][t], key=lambda e: e["seed"])] \
        if "targets_order" in m4 else [e["remaining"] for t in m4["configs"][k]["episodes"] for e in sorted(m4["configs"][k]["episodes"][t], key=lambda e: e["seed"])]
    L += ["", "## Paired comparison with the pure learned planner", "",
          "Each episode is paired with the `never` episode on the same target with the same planner seed (the random stream is the same: "
          "every trigger draws one uniform number per decision). Difference of the share of the initial error left (negative = better than never).", "",
          "| planner | difference [percentage points] | p | oracle rollouts per episode |", "|---|---|---|---|"]
    for k, v in m4["configs"].items():
        if k != "never":
            d, h, pv, n = paired(rem4(k), rem4("never"))
            L.append(f"| {k} | {100 * d:+.1f} ± {100 * h:.1f} | {pv:.3f} | {v['summary']['calls_per_episode'][0]:.1f} |")
    heads = [(f"{a} q{q}", f"random p{p}") for a in ("misrank", "disagreement") for q, p in ((25, 75), (50, 50), (75, 25))] + \
            [(f"misrank q{q}", f"disagreement q{q}") for q in (25, 50, 75)]
    heads = [(a, b) for a, b in heads if a in m4["configs"] and b in m4["configs"]]
    if heads:
        L += ["", "## Head to head at the same firing rate", "",
              "A trigger at its q-quantile threshold fires at about 1 - q of the decisions, so `q25` is compared with `random p75`. "
              "Paired as above; p from the paired t-test and from the Wilcoxon signed-rank test.", "",
              "| comparison | difference [percentage points] | p (t) | p (Wilcoxon) |", "|---|---|---|---|"]
        for a, b in heads:
            d, h, pv, n = paired(rem4(a), rem4(b))
            pw = float(stats.wilcoxon(rem4(a), rem4(b)).pvalue)
            L.append(f"| {a} - {b} | {100 * d:+.2f} ± {100 * h:.2f} | {pv:.1g} | {pw:.1g} |")
        L += ["", "Share of decisions at which the trigger fired, by target:", "",
              "| planner | " + " | ".join(next(iter(m4["configs"].values()))["episodes"]) + " |",
              "|---|" + "---|" * len(next(iter(m4["configs"].values()))["episodes"])]
        for k, v in m4["configs"].items():
            if k not in ("never", "always"):
                L.append(f"| {k} | " + " | ".join(f"{100 * np.mean([i['fired'] for e in v['episodes'][t] for i in e['info']]):.0f} %"
                                                    for t in v["episodes"]) + " |")
    if "signal" in m4:
        sg = m4["signal"]
        L += ["", "## Which signal predicts what a re-ranking gains?", "",
              f"At each of the {sg['n']} decisions of `always`, the gain of the re-ranking is the true cost of the action the ensemble ranked first "
              f"minus the true cost of the action executed (>= 0; mean {sg['mean_gain_mm']:.3f} mm). "
              "Signals: `spread` (disagreement about the map of the chosen action), `regret` (members' expected regret), "
              "`misrank` (probability that another of the top candidates is truly better).", "",
              "| signal | Spearman with the gain | share of the total gain in the quarter of decisions with the largest signal |", "|---|---|---|"]
        for k in sg["spearman"]:
            L.append(f"| {k} | {sg['spearman'][k]:+.2f} | {100 * sg['share_of_gain_in_top_quarter'][k]:.0f} % |")
        if "at_decisions" in m4:
            L += ["", f"At the decisions of `never`, all members rank the chosen action first in {100 * m4['at_decisions']['share_regret_zero']:.0f} % of cases (regret exactly 0)."]
    L += ["", "![Pareto](m4_fallback.png)"]
    write("M4.md", L)
