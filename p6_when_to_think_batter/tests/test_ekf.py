"""EKF: accuracy, consistency and reaction to a surprise."""
import numpy as np

from batter import config
from batter.estim.ekf import BallEKF, crossing_batch
from batter.estim.flight import FlightModel
from batter.models.pitch import generate
from batter.models.scene import build_flight_model


def _run(n=300, seed=0, surprise="none", adaptive=True):
    """Rows (time to crossing, crossing-point error, Mahalanobis^2, NIS) for every world and step."""
    cfg = config.load()
    fm = FlightModel.identify(build_flight_model(cfg["sim"], cfg["ball"]))
    P = generate(fm, cfg, n, seed, surprise=surprise)
    ekf = BallEKF(fm, n, cfg["measurement"])
    ekf.prm.adaptive = adaptive
    rng = np.random.default_rng(seed)
    x = P.x0.copy()
    truth = [x.copy()]
    rows = []
    for k in range(int(P.cross_t.max() / fm.h) + 1):
        chg = (P.s_kind == 1) & (P.s_step == k)
        x[chg, 6:9] = P.s_spin[chg]
        gust = (P.s_kind == 2) & (k >= P.s_step) & (k < P.s_step + P.s_len)
        x = fm.step(x, np.where(gust[:, None], P.s_force / cfg["ball"]["mass"], 0.0))
        truth.append(x.copy())
        j = k + 1 - ekf.nd                      # the camera shows the ball n_d steps ago
        if j < 0:
            continue
        z = truth[j][:, :3] + rng.normal(size=(n, 3)) * ekf.meas_sigma(truth[j][:, :3])[:, None]
        ekf.step(z, np.ones(n, bool))
        traj, Ppp = ekf.forecast(90)
        c = crossing_batch(traj, Ppp, cfg["pitch"]["strike_x"])
        tr = fm.rollout(x, 90)
        ct = crossing_batch(tr, np.zeros(tr.shape[:2] + (3, 3)), cfg["pitch"]["strike_x"])
        e = (c["p"] - ct["p"])[:, 1:]
        m2 = np.einsum("bi,bij,bj->b", e, np.linalg.inv(c["cov"] + 1e-12 * np.eye(2)), e)
        tau = (c["steps"] - ekf.nd) * fm.h
        ok = c["found"] & ct["found"] & (tau > 0) & (ekf.n_fix > 8)
        rows.append(np.stack([tau, np.linalg.norm(e, axis=1), m2, ekf.nis], axis=1)[ok])
    return np.concatenate(rows), ekf


def _sel(R, lo, hi):
    return R[(R[:, 0] >= lo) & (R[:, 0] < hi)]


def test_crossing_point_converges():
    R, _ = _run()
    assert np.median(_sel(R, 0.3, 0.4)[:, 1]) > 0.03       # early: spin is unknown, centimetres off
    assert np.median(_sel(R, 0.0, 0.05)[:, 1]) < 0.008     # at the plate: millimetres


def test_filter_is_consistent():
    """NIS has mean 3 (chi-square, 3 dof) and the crossing-point covariance is calibrated."""
    R, ekf = _run()
    assert 2.7 < R[:, 3].mean() < 3.3
    cover = (_sel(R, 0.1, 0.4)[:, 2] < 4.605).mean()       # 90 % ellipse of a 2-d Gaussian
    assert 0.85 < cover < 0.98
    assert (ekf.n_reset > 0).mean() < 0.03                 # and the surprise detector does not cry wolf


def test_covariance_reset_on_a_gust():
    """A gust trips the NIS detector in most pitches; the reset makes the covariance less
    over-confident.  (It does not make the estimate itself better: by the time a gust is
    visible above the camera noise there are few frames left.)"""
    Ra, ekf_a = _run(surprise="gust", adaptive=True)
    Rf, _ = _run(surprise="gust", adaptive=False)
    assert (ekf_a.n_reset > 0).mean() > 0.5
    cover = lambda R: (_sel(R, 0.0, 0.15)[:, 2] < 4.605).mean()
    assert cover(Ra) > cover(Rf) + 0.05
