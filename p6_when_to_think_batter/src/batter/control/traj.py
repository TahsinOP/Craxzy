"""Joint-space swing trajectories as QPs (closed form, OSQP when a limit is active).

A swing has three phases, all written in the simulator's integrator so the reference is
exactly trackable (v_{k+1} = v_k + h a_k, q_{k+1} = q_k + h v_{k+1}):

  approach   N steps from the current joint state (q0, v0) to (q_s, v_s): the strike
             configuration and the strike joint velocity.  Each joint solves

                 min  sum_k a_k^2 + w_j sum_k (a_{k+1} - a_k)^2
                 s.t. q_N = q_s,  v_N = v_s,  q_min <= q_k <= q_max, |v_k| <= v_max, |a_k| <= a_max.

  through    constant joint velocity v_s across the predicted impact, so that a timing
             error of a step moves the contact along the swing but does not change the
             bat's velocity;
  brake      decelerate to rest at the acceleration limit.

Per joint: the equality-constrained minimiser is a banded linear solve; if it satisfies
the inequalities it is the QP optimum, otherwise OSQP solves the full problem.  If that
is infeasible (the strike is too far away for the time left) the joint solves the *soft*
problem instead: the same limits, the terminal conditions as a quadratic penalty,

    min  rho_q (q_N - q_s)^2 + rho_v (v_N - v_s)^2 + eps * effort,

i.e. it gets as close to the strike as the limits allow.  The remaining terminal miss is
returned so the planner can weigh it.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solveh_banded

from batter.models.scene import QD_MAX, QDD_MAX

_SOLVED = 1  # osqp status_val for "solved"


@dataclass
class Plan:
    """Joint reference sampled at the control step: q (K+1,7), v (K+1,7), a (K,7)."""

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    k0: int                       # simulation step index at which the plan starts
    info: dict = field(default_factory=dict)

    def ref(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q_ref, v_ref, a_ref) at simulation step k; holds the final state afterwards."""
        i = k - self.k0
        if i >= self.a.shape[0]:
            return self.q[-1], np.zeros(7), np.zeros(7)
        i = max(i, 0)
        return self.q[i], self.v[i], self.a[i]


class SwingQP:
    def __init__(self, h: float, q_min: np.ndarray, q_max: np.ndarray, v_max: np.ndarray = QD_MAX,
                 a_max: np.ndarray = QDD_MAX, w_jerk: float = 0.5, limit_margin: float = 0.01,
                 vel_scale: float = 0.97, acc_scale: float = 0.9, osqp_max_iter: int = 600, cache_size: int = 64,
                 rho_q: float = 1e4, rho_v: float = 1e2, soft_eps: float = 1e-4, early_weight: float = 0.0):
        self.h = h
        self.q_min, self.q_max = q_min + limit_margin, q_max - limit_margin
        self.v_max, self.a_max = vel_scale * v_max, acc_scale * a_max
        self.w_jerk = w_jerk
        self.osqp_max_iter = osqp_max_iter
        self._cf: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._osqp: OrderedDict[int, osqp.OSQP] = OrderedDict()
        self._soft: OrderedDict[int, osqp.OSQP] = OrderedDict()
        self.rho_q, self.rho_v, self.soft_eps = rho_q, rho_v, soft_eps
        self.early_weight = early_weight
        self._cache_size = cache_size
        self.n_osqp = 0
        self.n_closed_form = 0

    # ---------------------------------------------------------------- matrices
    def terminal_rows(self, N: int) -> np.ndarray:
        """E (2, N): E a = (v_N - v_0, q_N - q_0 - N h v_0)."""
        h = self.h
        return np.vstack([np.full(N, h), h * h * np.arange(N, 0, -1.0)])

    def accel_weights(self, N: int) -> np.ndarray:
        """Weight of each step's acceleration in the cost: 1, plus ``early_weight`` (1 - k/N)^2.

        With a positive ``early_weight`` the arm moves little early in the horizon and most of the
        way late: while the forecast it is aiming at is still poor, it keeps its options open
        (used by the rally, where a swing committed to the first forecast often cannot be
        corrected in time)."""
        return 1.0 + self.early_weight * (1.0 - np.arange(N) / max(N, 1)) ** 2

    def cost_matrix(self, N: int) -> np.ndarray:
        Dj = np.diff(np.eye(N), axis=0)
        return 2.0 * (np.diag(self.accel_weights(N)) + self.w_jerk * Dj.T @ Dj)

    def _closed_form_factors(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        f = self._cf.get(N)
        if f is None:
            w = self.w_jerk
            diag = np.full(N, 2.0 * 2.0 * w)
            diag[0] = diag[-1] = 2.0 * w
            if N == 1:
                diag[0] = 0.0
            diag = diag + 2.0 * self.accel_weights(N)
            ab = np.vstack([np.r_[0.0, np.full(N - 1, -2.0 * w)], diag])
            PiEt = solveh_banded(ab, self.terminal_rows(N).T)
            f = (PiEt, np.linalg.inv(self.terminal_rows(N) @ PiEt))
            self._cf[N] = f
        return f

    def closed_form(self, q0: np.ndarray, v0: np.ndarray, qs: np.ndarray, vs: np.ndarray, N: int) -> np.ndarray:
        """Minimisers subject to the terminal conditions only, all joints at once: (N, J)."""
        PiEt, Sinv = self._closed_form_factors(N)
        b = np.stack([vs - v0, qs - q0 - N * self.h * v0])           # (2, J)
        return PiEt @ (Sinv @ b)

    def violation(self, a: np.ndarray, q0: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """Per joint, the largest relative excess over the acceleration/velocity/position limits (<= 0: fine)."""
        v = v0 + self.h * np.cumsum(a, axis=0)
        q = q0 + self.h * np.cumsum(v, axis=0)
        return np.max(np.stack([np.abs(a).max(0) / self.a_max - 1.0, np.abs(v).max(0) / self.v_max - 1.0,
                                (self.q_min - q.min(0)) / 0.1, (q.max(0) - self.q_max) / 0.1]), axis=0)

    def _osqp_problem(self, N: int) -> osqp.OSQP:
        s = self._osqp.get(N)
        if s is not None:
            self._osqp.move_to_end(N)
            return s
        h = self.h
        L = np.tril(np.ones((N, N)))
        C = np.cumsum(L, axis=0)
        A = np.vstack([np.eye(N), h * L, h * h * C, self.terminal_rows(N)])
        s = osqp.OSQP()
        s.setup(P=sp.csc_matrix(np.triu(self.cost_matrix(N))), q=np.zeros(N), A=sp.csc_matrix(A), l=np.zeros(A.shape[0]),
                u=np.zeros(A.shape[0]), verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=self.osqp_max_iter,
                polishing=False, warm_starting=False)
        self._osqp[N] = s
        if len(self._osqp) > self._cache_size:
            self._osqp.popitem(last=False)
        return s

    def solve_osqp(self, j: int, q0: float, v0: float, qs: float, vs: float, N: int):
        k = np.arange(1, N + 1)
        drift = q0 + k * self.h * v0
        b = np.array([vs - v0, qs - drift[-1]])
        l = np.concatenate([np.full(N, -self.a_max[j]), np.full(N, -self.v_max[j] - v0), self.q_min[j] - drift, b])
        u = np.concatenate([np.full(N, self.a_max[j]), np.full(N, self.v_max[j] - v0), self.q_max[j] - drift, b])
        s = self._osqp_problem(N)
        s.update(l=l, u=u)
        res = s.solve(raise_error=False)
        self.n_osqp += 1
        ok = res.info.status_val == _SOLVED and res.x is not None and bool(np.all(np.isfinite(res.x)))
        return res.x if ok else None

    def _soft_problem(self, N: int) -> osqp.OSQP:
        s = self._soft.get(N)
        if s is not None:
            self._soft.move_to_end(N)
            return s
        h = self.h
        L = np.tril(np.ones((N, N)))
        C = np.cumsum(L, axis=0)
        A = np.vstack([np.eye(N), h * L, h * h * C])
        E = self.terminal_rows(N)
        Pm = self.soft_eps * self.cost_matrix(N) + 2.0 * E.T @ np.diag([self.rho_v, self.rho_q]) @ E
        s = osqp.OSQP()
        s.setup(P=sp.csc_matrix(np.triu(Pm)), q=np.zeros(N), A=sp.csc_matrix(A), l=np.zeros(A.shape[0]),
                u=np.zeros(A.shape[0]), verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=self.osqp_max_iter,
                polishing=False, warm_starting=False)
        self._soft[N] = s
        if len(self._soft) > self._cache_size:
            self._soft.popitem(last=False)
        return s

    def solve_soft(self, j: int, q0: float, v0: float, qs: float, vs: float, N: int) -> np.ndarray:
        """Closest approach to (qs, vs) within the limits."""
        k = np.arange(1, N + 1)
        drift = q0 + k * self.h * v0
        b = np.array([vs - v0, qs - drift[-1]])
        E = self.terminal_rows(N)
        l = np.concatenate([np.full(N, -self.a_max[j]), np.full(N, -self.v_max[j] - v0), self.q_min[j] - drift])
        u = np.concatenate([np.full(N, self.a_max[j]), np.full(N, self.v_max[j] - v0), self.q_max[j] - drift])
        s = self._soft_problem(N)
        s.update(q=-2.0 * E.T @ (np.array([self.rho_v, self.rho_q]) * b), l=l, u=u)
        res = s.solve(raise_error=False)
        self.n_osqp += 1
        if res.x is None or not np.all(np.isfinite(res.x)):
            return np.clip(self.closed_form(np.array([q0]), np.array([v0]), np.array([qs]), np.array([vs]), N)[:, 0],
                           -self.a_max[j], self.a_max[j])
        return np.clip(res.x, -self.a_max[j], self.a_max[j])

    # ---------------------------------------------------------------- approach
    def approach(self, q0: np.ndarray, v0: np.ndarray, qs: np.ndarray, vs: np.ndarray, N: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Accelerations (N, 7) towards (qs, vs) in N steps.

        Returns (a, terminal position miss (7,), terminal velocity miss (7,), effort).
        """
        N = max(int(N), 2)
        a = self.closed_form(q0, v0, qs, vs, N)
        viol = self.violation(a, q0, v0)
        for j in np.flatnonzero(viol > 1e-9):
            aj = self.solve_osqp(j, float(q0[j]), float(v0[j]), float(qs[j]), float(vs[j]), N) if N >= 2 else None
            if aj is None:
                aj = self.solve_soft(j, float(q0[j]), float(v0[j]), float(qs[j]), float(vs[j]), N)
            a[:, j] = aj
        self.n_closed_form += int((viol <= 1e-9).sum())
        vN = v0 + self.h * a.sum(axis=0)
        qN = q0 + self.h * np.cumsum(v0 + self.h * np.cumsum(a, axis=0), axis=0)[-1]
        effort = float(np.sum((a / self.a_max) ** 2) * self.h)
        return a, qN - qs, vN - vs, effort

    def brake(self, v: np.ndarray) -> np.ndarray:
        """Decelerate every joint to rest at its acceleration limit: (K, 7)."""
        n = np.ceil(np.abs(v) / (self.a_max * self.h) - 1e-9).astype(int)
        K = max(int(n.max()), 1)
        a = np.zeros((K, 7))
        for j in range(7):
            if n[j] > 0:
                a[: n[j], j] = -v[j] / (n[j] * self.h)
        return a

    def swing(self, q0: np.ndarray, v0: np.ndarray, q_imp: np.ndarray, v_imp: np.ndarray, steps_to_impact: float,
              k0: int = 0, n_pre: int = 1, n_post: int = 2) -> tuple[Plan, np.ndarray, np.ndarray, float]:
        """Full swing: be at q_imp with velocity v_imp ``steps_to_impact`` (fractional) steps from now.

        The constant-velocity phase starts ``n_pre`` steps before the impact step and lasts
        until ``n_post`` steps after it (less of it if the impact is very close).  Returns
        (plan, position miss (7,), velocity miss (7,), effort); the misses are zero when the
        strike is reachable within the limits.
        """
        N = max(int(np.floor(steps_to_impact)) - n_pre, 2)
        qs = q_imp + (N - steps_to_impact) * self.h * v_imp          # on the constant-velocity line through the impact
        a1, miss_q, miss_v, effort = self.approach(q0, v0, qs, v_imp, N)
        v_end = v0 + self.h * a1.sum(axis=0)
        A = np.vstack([a1, np.zeros((n_pre + n_post + 1, 7)), self.brake(v_end)])
        V = np.vstack([v0, v0 + self.h * np.cumsum(A, axis=0)])
        Q = np.vstack([q0, q0 + self.h * np.cumsum(V[1:], axis=0)])
        return Plan(q=Q, v=V, a=A, k0=k0), miss_q, miss_v, effort
