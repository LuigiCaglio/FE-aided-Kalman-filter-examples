# Author : Luigi Caglio (lucag@dtu.dk)
# Date   : 2026-11-23
# Repo   : https://github.com/LuigiCaglio/FE-aided-Kalman-filter-examples
# Refs.  : [1] L. Caglio, H. Stang, R. Brincker, E. Katsanos (2023). FE-aided Kalman Filter
#              for nonlinear state estimation with unknown input. Mechanical Systems and
#              Signal Processing, 200, 110513. https://doi.org/10.1016/j.ymssp.2023.110513
#          [2] L. Caglio, H. Stang, E. Katsanos (2025). Bayesian model updating and nonlinear
#              response estimation of structures subjected to unknown loads via FE-aided
#              Kalman Filtering. Mechanical Systems and Signal Processing, 241, 113505.
#              https://doi.org/10.1016/j.ymssp.2025.113505
# License: MIT
"""Bayesian model updating with the FE-aided AEKF: MAP estimate + Laplace approximation.

Parameters are handled in normalised form x = theta / theta_ref, so that one step size and
one tolerance suit all of them.
"""

import numpy as np
from scipy.optimize import minimize


class NegLogPosterior:
    """phi(theta) = -ln p(d_1:N | theta) - ln p(theta), eq. (34) of [2], uniform prior on bounds.

    `nll(theta)` must return the negative log-likelihood (e.g. from run_aekf). Every
    evaluation is stored in `history` (theta, phi), so it can be plotted afterwards.
    """

    def __init__(self, nll, theta_ref, bounds):
        self.nll, self.theta_ref = nll, np.asarray(theta_ref, float)
        self.bounds = np.asarray(bounds, float)          # (n_par, 2), physical units
        self.history, self._cache = [], {}

    def __call__(self, x):
        theta = np.asarray(x) * self.theta_ref
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        tol = 1e-9 * (hi - lo)                           # round-off tolerance at the edges
        if np.any(theta < lo - tol) or np.any(theta > hi + tol):
            return np.inf                                # outside the uniform prior
        theta = np.clip(theta, lo, hi)
        key = tuple(np.round(theta, 6))
        if key not in self._cache:
            self._cache[key] = self.nll(theta)           # - ln prior = const -> omitted
            self.history.append((theta, self._cache[key]))
            print(f"  eval {len(self.history):3d}: theta = {fmt(theta)}  phi = {self._cache[key]:.2f}")
        return self._cache[key]


def fmt(theta):
    return "[" + ", ".join(f"{v:.4g}" for v in theta) + "]"


def map_estimate(phi, theta0, xatol=1e-3, fatol=10.0, maxiter=60):
    """MAP estimate, eq. (36): minimise phi with Nelder-Mead (derivative-free).

    fatol must be larger than the small-scale noise of phi (jumps of a few tens caused by
    yield events and by the stiffness-update threshold), otherwise the simplex never stops.
    xatol is relative (parameters are normalised by theta_ref).
    """
    x0 = np.asarray(theta0, float) / phi.theta_ref
    bounds = phi.bounds / phi.theta_ref[:, None]
    res = minimize(phi, x0, method="Nelder-Mead", bounds=bounds,
                   options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter))
    return res.x * phi.theta_ref, res


class Surrogate:
    """GP (Kriging) surrogate of phi, with the same interface as NegLogPosterior."""

    def __init__(self, gpr, theta_ref, bounds):
        self.gpr, self.theta_ref, self.bounds = gpr, theta_ref, bounds

    def to_unit(self, theta):
        return (theta - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def __call__(self, x):
        u = self.to_unit(np.asarray(x) * self.theta_ref)
        return float(self.gpr.predict_values(u[None, :])[0, 0])


def map_estimate_ego(phi, n_doe=8, n_iter=15, seed=0):
    """MAP estimate, eq. (36), by Bayesian optimisation (EGO, Expected Improvement) with SMT.

    phi is evaluated at an initial Latin-hypercube design (n_doe points) and at n_iter infill
    points; the MAP is then refined on the Kriging surrogate. Returns theta_MAP and the
    surrogate (used for the Hessian, as in [2]).
    """
    from smt.applications import EGO
    from smt.surrogate_models import KRG
    from smt.design_space import DesignSpace
    from smt.sampling_methods import LHS

    n = len(phi.theta_ref)
    lo, hi = phi.bounds[:, 0], phi.bounds[:, 1]
    unit = np.array([[0.0, 1.0]] * n)                    # GP works on the unit box

    def fun(U):                                          # U: (m, n) points in the unit box
        out = []
        for u in np.clip(U, 0.0, 1.0):
            theta = np.clip(lo + u * (hi - lo), lo, hi)  # avoid round-off just outside the box
            f = phi(theta / phi.theta_ref)
            if not np.isfinite(f):                       # failed AEKF run: finite penalty for the GP
                f = max(v for _, v in phi.history if np.isfinite(v)) if phi.history else 1e12
            out.append([f])
        return np.array(out)

    xdoe = LHS(xlimits=unit, criterion="ese", seed=seed)(n_doe)
    ego = EGO(n_iter=n_iter, criterion="EI", xdoe=xdoe, seed=seed,
              surrogate=KRG(design_space=DesignSpace(unit), print_global=False))
    u_best, _, _, _, _ = ego.optimize(fun=fun)

    sm = Surrogate(ego.gpr, phi.theta_ref, phi.bounds)
    res = minimize(lambda u: float(ego.gpr.predict_values(np.asarray(u)[None, :])[0, 0]),
                   u_best, method="L-BFGS-B", bounds=unit)
    return lo + res.x * (hi - lo), sm


def hessian_fd(phi, theta, h=0.01):
    """Hessian of phi at theta (physical units) by central finite differences.

    h is the relative step (a fraction of each parameter); a step of about 1 % keeps small
    kinks of phi (yield events, stiffness-update threshold) from dominating the result.
    """
    x = np.asarray(theta, float) / phi.theta_ref
    n, Hx = len(x), np.zeros((len(x), len(x)))
    f0 = phi(x)
    step = h * x
    for i in range(n):
        ei = np.zeros(n); ei[i] = step[i]
        Hx[i, i] = (phi(x + ei) - 2 * f0 + phi(x - ei)) / step[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = step[j]
            Hx[i, j] = Hx[j, i] = (phi(x + ei + ej) - phi(x + ei - ej)
                                   - phi(x - ei + ej) + phi(x - ei - ej)) / (4 * step[i] * step[j])
    T = np.diag(1 / phi.theta_ref)                       # d x / d theta
    return T @ Hx @ T                                    # Hessian w.r.t. theta


def laplace(H):
    """Laplace approximation, eqs. (37)-(38): posterior covariance = H^-1."""
    cov = np.linalg.inv(H)
    if np.any(np.linalg.eigvalsh(cov) <= 0):
        print("WARNING: Hessian not positive definite at the MAP (flat or kinked phi)")
    return cov
