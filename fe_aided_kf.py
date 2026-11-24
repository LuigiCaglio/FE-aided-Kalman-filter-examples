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
"""The steps of main_FEaidedKF.py as functions, for repeated use in main_FEaidedKF_BayesianUpdating.py.

true_response : nonlinear OpenSees transient analysis of the frame (the "truth")
measurements  : noisy absolute accelerations d_k and their covariance R
run_aekf      : FE-aided AEKF for given model parameters (E, fy), incl. the negative
                log-likelihood of the measurements, eq. (33) of [2]
"""

import numpy as np
import openseespy.opensees as ops
from scipy.linalg import block_diag
from tqdm import tqdm

from frame_2storey_2bay_2D import (build_frame, add_gravity_pattern, gravity_loads,
                                   beam_equivalent_loads)
from ops_helpers import (read_at2, system_matrices, rayleigh_damping, run_transient,
                         tangent_update, section_response, static_analysis, load_vector,
                         nodal_vector)

G = 9.81


def _model_with_gravity(E, fy, gravity):
    """Build the frame, apply gravity statically, define Rayleigh damping."""
    build_frame(E=E, fy=fy)
    if gravity:
        add_gravity_pattern(gravity, g=G)
        static_analysis()
    return rayleigh_damping()


def true_response(record, scale, E, fy, gravity="distributed", sections=()):
    """Nonlinear transient analysis of the frame under the scaled ground motion."""
    acc_g, dt = read_at2(record)
    _model_with_gravity(E, fy, gravity)
    M, C, K, dofs = system_matrices()
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *acc_g, "-factor", G * scale, "-prependZero")
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    Y, Yd, Ydd, ug_dd, MK, _ = run_transient(dt, acc_g.size, dofs, sections=list(sections))
    ops.wipe()
    return dict(Y=Y, Yd=Yd, Ydd=Ydd, ug_dd=ug_dd, MK=MK, dofs=dofs, dt=dt, n_steps=acc_g.size)


def measurements(true, sensor_nodes, noise, seed=2):
    """Absolute accelerations at the sensor nodes + Gaussian noise (std = noise * signal std)."""
    sens = [true["dofs"][nd][0] for nd in sensor_nodes]
    ydd_abs = true["Ydd"][:, sens] + true["ug_dd"][:, None]
    rng = np.random.RandomState(seed)
    noise_std = noise * ydd_abs.std(axis=0)
    d = np.empty_like(ydd_abs)
    for i in range(len(sens)):
        d[:, i] = ydd_abs[:, i] + rng.normal(scale=noise_std[i], size=ydd_abs.shape[0])
    return d, np.diag(noise_std**2)


def run_aekf(d, R, dt, sensor_nodes, E, fy, gravity="distributed", s_load=1e-2, q_state=1e-20,
             k_update_tol=5e-3, sections=(), gamma=0.5, beta=0.25, progress=True):
    """FE-aided AEKF (same algorithm as main_FEaidedKF.py) for the parameters (E, fy).

    Returns a dict with the estimates za, the diagonal of P, the estimated moment-curvature
    at `sections`, and nll = -ln p(d_1:N | theta), eq. (33) of [2] (summed over k).
    """
    n_steps = d.shape[0] - 1                      # as in main_FEaidedKF.py (last sample unused)
    a_r, b_r = _model_with_gravity(E, fy, gravity)
    M, C, K, dofs = system_matrices()
    n = M.shape[0]
    inv_M = np.linalg.inv(M)

    if gravity == "nodal":
        V = load_vector(gravity_loads(G), dofs)
    elif gravity == "distributed":
        V = load_vector(beam_equivalent_loads(G), dofs)
    else:
        V = np.zeros(n)

    hor = [dofs[nd][0] for nd in dofs if dofs[nd][0] >= 0]
    sens = [dofs[nd][0] for nd in sensor_nodes]
    S_g = np.zeros((n, 1)); S_g[hor, 0] = 1.0
    S_p = -M @ S_g
    S_a = np.eye(n)[sens]
    n_p = S_p.shape[1]
    S = s_load * np.eye(n_p)
    Q = q_state * np.eye(2 * n)

    a1, a2 = (1 - gamma) * dt, dt**2 / 2 * (1 - 2 * beta)
    a3, a4 = gamma * dt, beta * dt**2
    O, I = np.zeros((n_p, 2 * n)), np.eye(n_p)

    def formulation(K):
        C = a_r * M + b_r * K
        H0 = np.block([[M + a4 * K, a4 * C, -a4 * S_p], [a3 * K, M + a3 * C, -a3 * S_p], [O, I]])
        H1 = np.block([[M - a2 * K, dt * M - a2 * C, a2 * S_p], [-a1 * K, M - a1 * C, a1 * S_p], [O, I]])
        H0inv = np.linalg.inv(H0)
        Aa = H0inv @ H1
        Qa = H0inv @ block_diag(np.zeros((2 * n, 2 * n)), S) @ H0inv.T + block_diag(Q, np.zeros((n_p, n_p)))
        Ha = np.hstack([-S_a @ inv_M @ K, -S_a @ inv_M @ C, np.zeros((len(sens), n_p))])
        return H0inv, Aa, Qa, Ha

    H3 = np.concatenate([(a2 + a4) * V, (a1 + a3) * V, np.zeros(n_p)])
    za = np.zeros((n_steps, 2 * n + n_p))
    za[0, :n] = nodal_vector(dofs, ops.nodeDisp)
    P_diag = np.zeros((n_steps, 2 * n + n_p))
    K_ref = K.copy()
    H0inv, Aa, Qa, Ha = formulation(K_ref)
    P = Qa.copy()                                 # initial covariance P_0
    P_R = V.copy()
    MK_est = np.zeros((n_steps, len(sections), 2))
    MK_est[0] = section_response(list(sections))
    nll, n_updates = 0.0, 0
    log2pi = len(sens) * np.log(2 * np.pi)

    steps = tqdm(range(1, n_steps), desc=f"AEKF E={E/1e9:.1f} GPa, fy={fy/1e6:.1f} MPa",
                 leave=False, disable=not progress)
    for k in steps:
        r = -P_R + K_ref @ za[k - 1, :n]
        H2 = np.concatenate([(a2 + a4) * r, (a1 + a3) * r, np.zeros(n_p)])
        za_minus = Aa @ za[k - 1] + H0inv @ (H2 + H3)
        P_minus = Aa @ P @ Aa.T + Qa
        E_k = S_a @ inv_M @ (r + V)
        S_k = Ha @ P_minus @ Ha.T + R
        e = d[k] - (Ha @ za_minus + E_k)
        G_k = np.linalg.solve(S_k.T, (P_minus @ Ha.T).T).T
        za[k] = za_minus + G_k @ e
        P = P_minus - G_k @ Ha @ P_minus
        P_diag[k] = np.diag(P)
        nll += 0.5 * (log2pi + np.linalg.slogdet(S_k)[1] + e @ np.linalg.solve(S_k, e))   # eq. (33)

        K, P_R, ok = tangent_update(za[k, :n], K, P_R, dofs, V)
        if ok != 0:                               # FE model cannot follow the estimate
            ops.wipe()
            return dict(za=za, P_diag=P_diag, MK_est=MK_est, nll=np.inf, n_updates=n_updates, dofs=dofs)
        if sections:
            MK_est[k] = section_response(list(sections))
        if np.max(np.abs((np.diag(K) - np.diag(K_ref)) / np.diag(K_ref))) > k_update_tol:
            K_ref = K.copy()
            H0inv, Aa, Qa, Ha = formulation(K_ref)
            n_updates += 1
    ops.wipe()
    return dict(za=za, P_diag=P_diag, MK_est=MK_est, nll=nll, n_updates=n_updates, dofs=dofs)
