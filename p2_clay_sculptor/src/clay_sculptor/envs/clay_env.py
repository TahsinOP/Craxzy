"""The clay bench: batched primitive rollouts on ``mujoco.batch`` (CPU, multi-thread).

A *state* is MuJoCo's full integration state vector (``mjSTATE_INTEGRATION``):
time, qpos, qvel, act (the servo integrators), **plastic** (the plastic strain of
every cage cell), warm start, controls and applied forces.  It is a flat float64
array; saving a state is copying it and restoring is writing it back.  The three
tool models have identical state layouts, so a state moves freely between tools.

``ClayEnv.rollout(states, tool, actions)`` executes one primitive per state in
parallel and returns the next states.  Before stepping, the arm part of each state
is set to the hover pose of its primitive at rest (see ``control.primitives`` for
why the free-space move between primitives is not simulated) and time is set to 0,
so the result depends only on (clay state, tool, action).
"""
from __future__ import annotations

import warnings

import mujoco
import numpy as np
from mujoco import batch as mjbatch

from clay_sculptor.control.arm import Kinematics
from clay_sculptor.control.primitives import Planner, PrimCfg
from clay_sculptor.envs.shape import Geometry, MapCfg
from clay_sculptor.models.scene import HOME_Q, TOOLS, ClayCfg, Index, SimCfg, build_model

STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


class StateLayout:
    """Slices of the components inside a state vector of spec ``spec``.

    MuJoCo concatenates the components in the order of their bits in ``mjtState``."""

    def __init__(self, m: mujoco.MjModel, spec: int = STATE_SPEC):
        self.spec, self.slices, adr = spec, {}, 0
        for i in range(int(mujoco.mjtState.mjNSTATE)):
            bit = 1 << i
            if spec & bit:
                n = mujoco.mj_stateSize(m, bit)
                self.slices[mujoco.mjtState(bit).name[len("mjSTATE_"):].lower()] = slice(adr, adr + n)
                adr += n
        self.size = adr
        assert adr == mujoco.mj_stateSize(m, spec)

    def __getitem__(self, name: str) -> slice:
        return self.slices[name]


def quiet_build(tool, clay, sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_model(tool, clay, sim)


class ClayEnv:
    def __init__(self, clay: ClayCfg = ClayCfg(), sim: SimCfg = SimCfg(), prim: PrimCfg = PrimCfg(),
                 mp: MapCfg = MapCfg(), nthread: int = 0, nsim: int = 64):
        self.clay, self.sim, self.prim = clay, sim, prim
        self.models = {t: quiet_build(t, clay, sim) for t in TOOLS}
        m = self.models["press"]
        self.ix = Index.of(m)
        self.layout = StateLayout(m)
        for t in TOOLS:   # identical layouts: a state of one tool model is a state of the others
            assert StateLayout(self.models[t]).slices == self.layout.slices
        self.geom = Geometry(m, clay, mp)
        self.kin = Kinematics(sim)
        self.planner = Planner(self.kin, clay, prim, sim.timestep)
        self.nthread, self.nsim = nthread, nsim
        self._batches: dict[str, mjbatch.Batch] = {}
        self._data = {t: mujoco.MjData(self.models[t]) for t in TOOLS}
        qs, vs, acts = self.layout["qpos"], self.layout["qvel"], self.layout["act"]
        self._arm_q = qs.start + self.ix.arm_qpos
        self._arm_v = vs.start + self.ix.arm_dof
        self._act = acts
        self.state0 = self._initial_state()
        self.oracle_calls = 0          # number of simulated primitives (the unit of oracle cost)

    # ---- single-world helpers ---------------------------------------------------
    def _initial_state(self) -> np.ndarray:
        """The undeformed block, settled under gravity, arm at home."""
        m, d = self.models["press"], self._data["press"]
        mujoco.mj_resetData(m, d)
        d.qpos[self.ix.arm_qpos] = HOME_Q
        d.ctrl[self.ix.arm_ctrl[:, 0]] = HOME_Q
        for _ in range(int(round(1.0 / m.opt.timestep))):
            mujoco.mj_step(m, d)
        d.time = 0.0
        return self.get_state(d)

    def get_state(self, d: mujoco.MjData, tool: str = "press") -> np.ndarray:
        s = np.empty(self.layout.size)
        mujoco.mj_getState(self.models[tool], d, s, STATE_SPEC)
        return s

    def set_state(self, d: mujoco.MjData, state: np.ndarray, tool: str = "press") -> None:
        mujoco.mj_setState(self.models[tool], d, np.ascontiguousarray(state, dtype=np.float64), STATE_SPEC)

    def nodes(self, states: np.ndarray) -> np.ndarray:
        """State(s) -> cage nodes ``(..., nx, ny, nz, 3)`` in clay coordinates."""
        return self.geom.nodes_from_qpos(np.asarray(states)[..., self.layout["qpos"]])

    def prepare(self, state: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
        """Copy of ``state`` with the arm at ``q_arm`` at rest, integrators cleared, time 0."""
        s = np.array(state, dtype=np.float64, copy=True)
        s[..., self._arm_q] = q_arm
        s[..., self._arm_v] = 0.0
        s[..., self._act] = 0.0
        s[..., self.layout["time"]] = 0.0
        s[..., self.layout["ctrl"]] = 0.0
        return s

    # ---- batched rollouts -------------------------------------------------------
    def _batch(self, tool: str) -> mjbatch.Batch:
        if tool not in self._batches:
            self._batches[tool] = mjbatch.Batch(self.models[tool], nsim=self.nsim, nthread=self.nthread)
        return self._batches[tool]

    def rollout(self, states: np.ndarray, tool: str, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Execute ``actions[i]`` with ``tool`` from ``states[i]``.

        ``states`` is ``(n, nstate)`` or a single ``(nstate,)`` state shared by all
        actions.  Returns ``(next_states (n, nstate), ok (n,) bool)``; ``ok`` is False
        where MuJoCo raised a warning (the state is then not to be trusted)."""
        actions = np.atleast_2d(np.asarray(actions, dtype=float))
        n = len(actions)
        states = np.broadcast_to(np.asarray(states, dtype=np.float64), (n, self.layout.size))
        refs = [self.planner.reference(tool, a) for a in actions]
        nstep = self.prim.nsteps(tool, self.sim.timestep)
        out = np.empty((n, self.layout.size))
        ok = np.ones(n, dtype=bool)
        b = self._batch(tool)
        for lo in range(0, n, self.nsim):
            hi = min(lo + self.nsim, n)
            k = hi - lo
            ids = np.arange(k)
            ctrl = np.stack([refs[i][1] for i in range(lo, hi)])
            assert ctrl.shape[1] == nstep
            b.state[:k] = self.prepare(states[lo:hi], np.stack([refs[i][0] for i in range(lo, hi)]))
            warn0 = np.array(b.warning[:k]).copy()
            b.rollout(nstep, control=ctrl, record=(mujoco.mjtState.mjSTATE_TIME,), ids=ids)
            out[lo:hi] = b.state[:k]
            ok[lo:hi] = (np.array(b.warning[:k]) == warn0).reshape(k, -1).all(axis=1)
        self.oracle_calls += n
        return out, ok

    def step(self, state: np.ndarray, tool: str, action: np.ndarray) -> np.ndarray:
        nxt, ok = self.rollout(state[None], tool, np.asarray(action)[None])
        if not ok[0]:
            raise RuntimeError(f"MuJoCo warning during {tool} {action}")
        return nxt[0]

    # ---- reference single-thread execution (tests, video, continuous mode) ------
    def step_reference(self, state: np.ndarray, tool: str, action: np.ndarray, on_step=None) -> np.ndarray:
        """Same as ``step`` with plain ``mj_step`` on one ``MjData`` (bit-identical to the batch)."""
        m, d = self.models[tool], self._data[tool]
        q0, ctrl = self.planner.reference(tool, np.asarray(action, dtype=float))
        self.set_state(d, self.prepare(state, q0), tool)
        for u in ctrl:
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
            if on_step is not None:
                on_step(m, d)
        return self.get_state(d, tool)

    def run_continuous(self, state: np.ndarray, tool: str, actions, on_step=None, t_transit: float = 0.5) -> np.ndarray:
        """Execute several primitives of one tool without resetting the arm: the
        free-space moves between primitives are simulated too.  Used for video and
        to measure what leaving them out of ``rollout`` costs."""
        m, d = self.models[tool], self._data[tool]
        actions = [np.asarray(a, dtype=float) for a in actions]
        q0, _ = self.planner.reference(tool, actions[0])
        self.set_state(d, self.prepare(state, q0), tool)
        prev = None
        for a in actions:
            if prev is not None:
                _, ctrl = self.planner.transit_reference(prev, a, t_transit)
                for u in ctrl:
                    d.ctrl[:] = u
                    mujoco.mj_step(m, d)
                    if on_step is not None:
                        on_step(m, d)
            _, ctrl = self.planner.reference(tool, a)
            for u in ctrl:
                d.ctrl[:] = u
                mujoco.mj_step(m, d)
                if on_step is not None:
                    on_step(m, d)
            prev = a
        return self.get_state(d, tool)
