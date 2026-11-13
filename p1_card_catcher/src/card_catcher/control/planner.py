"""Interception planner: joint-space trajectory optimisation as a QP (OSQP).

Given the current joint state (q0, v0), a goal configuration q_g (from IK on the
predicted catch pose) and N control steps until the object arrives, each joint
solves a discrete minimum-effort problem in its accelerations a_0..a_{N-1}:

    min   sum_k a_k^2 + w_j sum_k (a_{k+1} - a_k)^2
    s.t.  v_k = v_0 + h sum_{i<k} a_i                   (the simulator's integrator,
          q_k = q_0 + k h v_0 + h^2 sum_{i<k} (k-i) a_i  so the plan is exactly trackable)
          q_min <= q_k <= q_max,  |v_k| <= v_max,  |a_k| <= a_max      (k = 1..N)
          q_N = q_g,  v_N = 0.

The states are eliminated (condensed form).  The seven joints are independent.

Solution strategy per joint
  1. Closed form.  Without the inequalities the problem is an equality-constrained
     least-squares problem, a = P^-1 E^T (E P^-1 E^T)^-1 b, with P tridiagonal (an
     O(N) banded solve).  If that solution satisfies every inequality it *is* the
     QP optimum (KKT with zero multipliers on the inactive bounds).
  2. OSQP, when some limit is active and the goal is still reachable in N steps.
     Reachability is decided analytically by the time-optimal (bang-coast-bang)
     transfer time of a double integrator, ``min_transfer_time``.
  3. Late.  If the goal is not reachable in N steps (or OSQP does not converge
     within its iteration budget), the joint takes the shortest horizon N2 > N, on
     a geometric grid, whose closed-form solution respects the limits.  It arrives
     late, at rest, as soon as a smooth minimum-effort motion allows.

Torque limits are represented by the datasheet acceleration limits here and by
saturation in the tracking controller; M3 reports how often torques saturate.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solveh_banded

from card_catcher.models.scene import QD_MAX, QDD_MAX

_SOLVED = 1  # osqp status_val for "solved"


@dataclass
class Plan:
    """Joint reference sampled at the control step: q (K+1,7), v (K+1,7), a (K,7)."""

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    k0: int                 # simulation step index at which the plan starts
    n_intercept: int        # steps from k0 to the planned interception
    on_time: np.ndarray     # (7,) bool: joint reaches its goal by the interception
    q_goal: np.ndarray

    def ref(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q_ref, v_ref, a_ref) at simulation step k; holds the final state afterwards."""
        i = k - self.k0
        if i >= self.a.shape[0]:
            return self.q[-1], np.zeros(7), np.zeros(7)
        i = max(i, 0)
        return self.q[i], self.v[i], self.a[i]

    @property
    def q_at_intercept(self) -> np.ndarray:
        return self.q[min(self.n_intercept, self.q.shape[0] - 1)]


def min_transfer_time(d: np.ndarray, v0: np.ndarray, v_max: np.ndarray, a_max: np.ndarray) -> np.ndarray:
    """Minimum time for a double integrator q'' = a, |a| <= a_max, |q'| <= v_max, to go
    from (0, v0) to rest at displacement d (elementwise; bang-coast-bang).

    With s the sign of (d - v0|v0|/(2 a_max)) (the side of the braking curve the goal
    is on), work in D = s d, u = s v0.  Accelerate at +a_max to the peak speed
    v_p = sqrt(a_max D + u^2/2), then brake: T = (2 v_p - u)/a_max.  If v_p > v_max,
    cruise at v_max: T = (2 v_max - u)/a_max + (D - (2 v_max^2 - u^2)/(2 a_max))/v_max.
    """
    d, v0 = np.asarray(d, float), np.asarray(v0, float)
    s = np.where(d - v0 * np.abs(v0) / (2 * a_max) >= 0, 1.0, -1.0)
    D, u = s * d, s * v0
    vp = np.sqrt(np.maximum(a_max * D + 0.5 * u * u, 0.0))
    t_tri = (2 * vp - u) / a_max
    d_ramp = (2 * v_max**2 - u * u) / (2 * a_max)
    t_trap = (2 * v_max - u) / a_max + (D - d_ramp) / v_max
    return np.where(vp <= v_max, t_tri, t_trap)


class TrajectoryQP:
    def __init__(self, h: float, q_min: np.ndarray, q_max: np.ndarray, v_max: np.ndarray = QD_MAX,
                 a_max: np.ndarray = QDD_MAX, w_jerk: float = 0.5, limit_margin: float = 0.01,
                 vel_scale: float = 0.95, acc_scale: float = 0.9, max_horizon_s: float = 2.5,
                 osqp_max_iter: int = 800, cache_size: int = 48):
        self.h = h
        self.q_min, self.q_max = q_min + limit_margin, q_max - limit_margin
        self.v_max, self.a_max = vel_scale * v_max, acc_scale * a_max
        self.w_jerk = w_jerk
        self.max_steps = int(max_horizon_s / h)
        self.osqp_max_iter = osqp_max_iter
        self._cf: dict[int, tuple[np.ndarray, np.ndarray]] = {}       # N -> (P^-1 E^T, S^-1)
        self._osqp: OrderedDict[int, osqp.OSQP] = OrderedDict()       # small LRU cache of OSQP problems
        self._cache_size = cache_size
        self.n_osqp = 0            # OSQP solves
        self.n_closed_form = 0     # joints solved in closed form

    # ---------------------------------------------------------------- matrices
    def terminal_rows(self, N: int) -> np.ndarray:
        """E (2, N): E a = (v_N - v_0, q_N - q_0 - N h v_0)."""
        h = self.h
        return np.vstack([np.full(N, h), h * h * np.arange(N, 0, -1.0)])

    def cost_matrix(self, N: int) -> np.ndarray:
        """P (N, N) with cost = 0.5 a^T P a = sum a_k^2 + w_j sum (a_{k+1} - a_k)^2 (tridiagonal)."""
        Dj = np.diff(np.eye(N), axis=0)
        return 2.0 * (np.eye(N) + self.w_jerk * Dj.T @ Dj)

    def _closed_form_factors(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        f = self._cf.get(N)
        if f is None:
            w = self.w_jerk
            diag = np.full(N, 2.0 * (1.0 + 2.0 * w))
            diag[0] = diag[-1] = 2.0 * (1.0 + w)
            if N == 1:
                diag[0] = 2.0
            ab = np.vstack([np.r_[0.0, np.full(N - 1, -2.0 * w)], diag])       # upper banded storage
            PiEt = solveh_banded(ab, self.terminal_rows(N).T)
            f = (PiEt, np.linalg.inv(self.terminal_rows(N) @ PiEt))
            self._cf[N] = f
        return f

    def closed_form(self, q0: float, v0: float, qg: float, N: int) -> np.ndarray:
        """Minimiser subject to the terminal conditions only."""
        PiEt, Sinv = self._closed_form_factors(N)
        return PiEt @ (Sinv @ np.array([-v0, qg - q0 - N * self.h * v0]))

    def within_limits(self, j: int, a: np.ndarray, q0: float, v0: float, tol: float = 1e-9) -> bool:
        v = v0 + self.h * np.cumsum(a)
        q = q0 + self.h * np.cumsum(v)
        return bool(np.abs(a).max() <= self.a_max[j] + tol and np.abs(v).max() <= self.v_max[j] + tol
                    and q.min() >= self.q_min[j] - tol and q.max() <= self.q_max[j] + tol)

    def _osqp_problem(self, N: int) -> osqp.OSQP:
        s = self._osqp.get(N)
        if s is not None:
            self._osqp.move_to_end(N)
            return s
        h = self.h
        L = np.tril(np.ones((N, N)))            # (h L a)_k   = v_k - v_0,              k = 1..N
        C = np.cumsum(L, axis=0)                # (h^2 C a)_k = q_k - q_0 - k h v_0
        A = np.vstack([np.eye(N), h * L, h * h * C, self.terminal_rows(N)])
        s = osqp.OSQP()
        s.setup(P=sp.csc_matrix(np.triu(self.cost_matrix(N))), q=np.zeros(N), A=sp.csc_matrix(A), l=np.zeros(A.shape[0]),
                u=np.zeros(A.shape[0]), verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=self.osqp_max_iter,
                polishing=True, warm_starting=False)
        self._osqp[N] = s
        if len(self._osqp) > self._cache_size:
            self._osqp.popitem(last=False)
        return s

    def solve_osqp(self, j: int, q0: float, v0: float, qg: float, N: int):
        """Full inequality-constrained problem; None unless OSQP reports "solved"."""
        k = np.arange(1, N + 1)
        drift = q0 + k * self.h * v0
        b = np.array([-v0, qg - drift[-1]])
        l = np.concatenate([np.full(N, -self.a_max[j]), np.full(N, -self.v_max[j] - v0), self.q_min[j] - drift, b])
        u = np.concatenate([np.full(N, self.a_max[j]), np.full(N, self.v_max[j] - v0), self.q_max[j] - drift, b])
        s = self._osqp_problem(N)
        s.update(l=l, u=u)
        res = s.solve(raise_error=False)
        self.n_osqp += 1
        ok = res.info.status_val == _SOLVED and res.x is not None and bool(np.all(np.isfinite(res.x)))
        return res.x if ok else None

    # ------------------------------------------------------------------- solve
    def _solve_joint(self, j: int, q0: float, v0: float, qg: float, N: int, n_min: int) -> tuple[np.ndarray, bool]:
        """Accelerations for one joint and whether it reaches the goal within N steps."""
        if n_min <= N:
            a = self.closed_form(q0, v0, qg, N)
            if self.within_limits(j, a, q0, v0):
                self.n_closed_form += 1
                return a, True
            a = self.solve_osqp(j, q0, v0, qg, N)
            if a is not None:
                return a, True
        # late: shortest horizon whose closed-form solution respects the limits
        N2 = max(N + 1, n_min)
        while N2 <= self.max_steps:
            a = self.closed_form(q0, v0, qg, N2)
            if self.within_limits(j, a, q0, v0):
                self.n_closed_form += 1
                return a, False
            N2 = int(np.ceil(1.08 * N2)) + 1
        return self._brake(j, v0), False

    def solve(self, q0: np.ndarray, v0: np.ndarray, q_goal: np.ndarray, N: int, k0: int = 0) -> Plan:
        """Plan from (q0, v0) to rest at q_goal, with the interception N >= 2 steps ahead."""
        N = int(np.clip(N, 2, self.max_steps))
        q_goal = np.clip(q_goal, self.q_min, self.q_max)
        t_min = min_transfer_time(q_goal - q0, v0, self.v_max, self.a_max)
        # +3 steps: the piecewise-constant discretisation reaches slightly less than continuous time
        n_min = np.ceil(t_min / self.h).astype(int) + 3
        acc, on_time = [], np.zeros(7, bool)
        for j in range(7):
            a, on_time[j] = self._solve_joint(j, float(q0[j]), float(v0[j]), float(q_goal[j]), N, int(n_min[j]))
            acc.append(a)
        K = max(len(a) for a in acc)
        A = np.zeros((K, 7))
        for j, a in enumerate(acc):
            A[: len(a), j] = a
        # integrate with the simulator's scheme so the plan obeys it to machine precision
        V = np.vstack([v0, v0 + self.h * np.cumsum(A, axis=0)])
        Q = np.vstack([q0, q0 + self.h * np.cumsum(V[1:], axis=0)])
        return Plan(q=Q, v=V, a=A, k0=k0, n_intercept=N, on_time=on_time, q_goal=q_goal)

    def _brake(self, j: int, v0: float) -> np.ndarray:
        a, v = [], v0
        while abs(v) > 1e-9 and len(a) < self.max_steps:
            ak = float(np.clip(-v / self.h, -self.a_max[j], self.a_max[j]))
            a.append(ak)
            v += self.h * ak
        return np.array(a if a else [0.0])
