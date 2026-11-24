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
"""FE-aided AEKF for a nonlinear steel frame, with acceleration AND strain measurements.

Joint estimation of the state (displacements, velocities) and the unknown input
(ground acceleration) of the 5-storey 3-bay steel frame of [1, 2], from noisy absolute
accelerations and strain gauges. The filter uses Newmark time integration; at every step the
OpenSees model is moved to the estimated displacements to obtain the tangent
stiffness and the restoring forces.

1. "True" response: nonlinear transient analysis in OpenSees.
2. Measurements: absolute accelerations at two nodes (1 % noise) and fibre strains at the
   strain gauges (5 % noise). The strain rows of the measurement equation follow Appendix B
   of [2]: eps_k ~ eps_FE,k-1 + H^s (y_k - y_k-1), with H^s from the FE model at every step.
3. AEKF on a fresh copy of the model.

Run:  python main_FEaidedKF_strainMeas_5storey.py -> figures/strainMeas_5storey_*.png
"""

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from scipy.linalg import block_diag
from tqdm import tqdm

from frame_5storey_3bay_2D import build_frame, add_gravity_pattern, gravity_loads
from ops_helpers import (read_at2, system_matrices, rayleigh_damping, run_transient,
                         tangent_update, section_response, static_analysis, load_vector,
                         nodal_vector, fiber_y, fiber_strain, strain_row,
                         plot_structure)

# Notation follows Caglio et al. (2025) [2] (equation numbers in the comments).
G = 9.81
RECORD = "synthetic_gm.at2"     # synthetic ground motion (PEER .at2 format, in g)
SCALE = 3.0                     # ground-motion scale factor
SENSOR_NODES = [23, 63,]         # horizontal absolute accelerations: 1st floor and roof
NOISE = 0.01                    # noise std / signal std (accelerations)
STRAIN_GAUGES = [(2021, 1, "bottom"),]   # (element, integration point, outer fibre): 1st-floor beam end
NOISE_STRAIN = 0.05             # noise std / signal std (strains)
S_LOAD = 1e-2                   # S: random-walk covariance of the input p [(m/s^2)^2], eq. (15)
Q_STATE = 1e-20                 # Q: process-noise variance on [y; y_dot], eq. (18)
GAMMA, BETA = 0.5, 0.25         # Newmark (average acceleration), eqs. (4)-(5)
K_UPDATE_TOL = 5e-3             # rebuild the formulation when max |dK_ii / K_ii| exceeds this (0 = every step)
GRAVITY = "nodal"               # gravity loads = lumped masses * g ("nodal") or None
ZETA = 0.02                     # Rayleigh damping ratio on modes 1 and 3, as in [2]
# moment-curvature output: (element, integration point); 5 Lobatto points, 1 = node i
MK_SECTIONS = {"column base, left (ele 1020)": (1020, 1),
               "column base, middle (ele 1121)": (1121, 1),
               "beam end, 1st floor (ele 2021)": (2021, 1),
               "beam end, 3rd floor (ele 4041)": (4041, 1)}


# =============================================================================
# 1. True response
# =============================================================================
acc_g, dt = read_at2(RECORD)
n_steps = acc_g.size

build_frame()
if GRAVITY:
    add_gravity_pattern(GRAVITY, g=G)
    static_analysis()                             # gravity first, then held constant
rayleigh_damping(ZETA)
M, C, K, dofs = system_matrices()
n = M.shape[0]

ops.timeSeries("Path", 1, "-dt", dt, "-values", *acc_g, "-factor", G * SCALE, "-prependZero")
ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
sections = list(MK_SECTIONS.values())
gauges = [(ele, ip, fiber_y(ele, ip, side)) for ele, ip, side in STRAIN_GAUGES]   # (ele, ip, y_fibre)
plot_structure("5-storey 3-bay steel frame (opsvis)", gauges=gauges, sensor_nodes=SENSOR_NODES)
Y, Yd, Ydd, ug_dd, MK_true, EPS_true = run_transient(dt, n_steps, dofs, sections=sections, gauges=gauges)
ops.wipe()

hor = [dofs[nd][0] for nd in dofs if dofs[nd][0] >= 0]  # horizontal DOFs
sens = [dofs[nd][0] for nd in SENSOR_NODES]
ydd_abs = Ydd[:, sens] + ug_dd[:, None]                  # absolute accelerations at the sensors

# =============================================================================
# 2. Measurements d_k = [absolute accelerations; strains] + noise v_k ~ N(0, R)
# =============================================================================
rng = np.random.RandomState(2)
clean = np.hstack([ydd_abs, EPS_true])
noise_std = np.concatenate([NOISE * ydd_abs.std(axis=0), NOISE_STRAIN * EPS_true.std(axis=0)])
d = clean + rng.normal(size=clean.shape) * noise_std
R = np.diag(noise_std**2)
n_s = len(gauges)                                 # number of strain gauges

# =============================================================================
# 3. FE-aided AEKF
# =============================================================================
build_frame()                                     # fresh model, used only by the filter
if GRAVITY:
    add_gravity_pattern(GRAVITY, g=G)
    static_analysis()
a_r, b_r = rayleigh_damping(ZETA)
M, C, K, dofs = system_matrices()
inv_M = np.linalg.inv(M)

# Known load V (gravity), eq. (19), as nodal loads on the free DOFs.
# V = the nodal gravity loads (lumped masses * g). V is also IMPLICITLY included in the FE
# model (the gravity pattern stays applied), so tangent_update applies only P_R - V as extra
# nodal loads, and the filter uses the physical P_R (= V at the start).
if GRAVITY == "nodal":
    V = load_vector(gravity_loads(G), dofs)
else:
    V = np.zeros(n)

S_g = np.zeros((n, 1)); S_g[hor, 0] = 1.0         # influence matrix, eq. (19)
S_p = -M @ S_g                                    # eq. (20); p_k = ground acceleration, eq. (21)
S_a = np.eye(n)[sens]                             # acceleration selection matrix, eq. (B.1)
n_p = S_p.shape[1]
S = S_LOAD * np.eye(n_p)
Q = Q_STATE * np.eye(2 * n)

a1, a2 = (1 - GAMMA) * dt, dt**2 / 2 * (1 - 2 * BETA)   # eq. (13)
a3, a4 = GAMMA * dt, BETA * dt**2
O, I = np.zeros((n_p, 2 * n)), np.eye(n_p)


def newmark_H0_H1(M, C, K):
    """Explicit Newmark form H0 z^a_{k+1} = H1 z^a_k + H2 + H3 + w^p, eqs. (6), (8), (9)."""
    H0 = np.block([[M + a4 * K, a4 * C, -a4 * S_p], [a3 * K, M + a3 * C, -a3 * S_p], [O, I]])
    H1 = np.block([[M - a2 * K, dt * M - a2 * C, a2 * S_p], [-a1 * K, M - a1 * C, a1 * S_p], [O, I]])
    return H0, H1


def process_noise(H0inv):
    """Q^a = H0^-1 [0 0; 0 S] H0^-T + [Q 0; 0 0], eq. (18)."""
    return H0inv @ block_diag(np.zeros((2 * n, 2 * n)), S) @ H0inv.T + block_diag(Q, np.zeros((n_p, n_p)))


# Known loads: S_q = I and q_k = V (gravity), eqs. (22)-(23), so H3 is constant, eq. (11)
H3 = np.concatenate([(a2 + a4) * V, (a1 + a3) * V, np.zeros(n_p)])

# z^a = [y; y_dot; p], eq. (7); initial state = static gravity state of the FE model
za = np.zeros((n_steps, 2 * n + n_p))
za[0, :n] = nodal_vector(dofs, ops.nodeDisp)
P_diag = np.zeros((n_steps, 2 * n + n_p))         # diagonal of P_k
P_R = V.copy()                                    # restoring forces (in equilibrium with V)
MK_est = np.zeros((n_steps, len(sections), 2))    # estimated [curvature, moment]
MK_est[0] = section_response(sections)
EPS_est = np.zeros((n_steps, n_s))                # estimated strains (FE model at the estimate)
EPS_est[0] = fiber_strain(gauges)


def formulation(K):
    """Matrices that depend on the tangent stiffness: H0^-1, A^a, Q^a, H^a."""
    C = a_r * M + b_r * K                         # Rayleigh damping with the tangent stiffness
    H0, H1 = newmark_H0_H1(M, C, K)
    H0inv = np.linalg.inv(H0)
    Aa = H0inv @ H1                               # eq. (14)
    Qa = process_noise(H0inv)                     # eq. (18)
    Ha = np.hstack([-S_a @ inv_M @ K, -S_a @ inv_M @ C, np.zeros((len(sens), n_p))])   # eq. (B.1)
    return H0inv, Aa, Qa, Ha


# The formulation is rebuilt only when the tangent stiffness has changed by more than
# K_UPDATE_TOL (Sec. 3.2 of [2]); the FE model itself is updated at every step.
K_ref = K.copy()
H0inv, Aa, Qa, Ha = formulation(K_ref)
P = Qa.copy()                                     # initial covariance P_0
n_updates = 0                                     # number of formulation rebuilds

# Main KF loop
for k in tqdm(range(1, n_steps), desc="FE-aided AEKF"):
    y_prev = za[k - 1, :n]
    r = -P_R + K_ref @ y_prev                     # -P_R,k + K_k y_k
    H2 = np.concatenate([(a2 + a4) * r, (a1 + a3) * r, np.zeros(n_p)])   # eq. (10)

    # prediction, eqs. (25)-(26)
    za_minus = Aa @ za[k - 1] + H0inv @ (H2 + H3)
    P_minus = Aa @ P @ Aa.T + Qa

    # measurement equation d = Ha z^a + E, eqs. (24), (B.1)-(B.2)
    # acceleration rows: Ha_acc (from the formulation), E = S_a M^-1 (-P_R + K y + V)
    # strain rows:       [H^s 0 0],  E = eps_FE,k-1 - H^s y_k-1  (FE model is at y_k-1)
    Hs = np.array([strain_row(ele, ip, y, dofs, n) for ele, ip, y in gauges]).reshape(n_s, n)
    Ha_k = np.vstack([Ha, np.hstack([Hs, np.zeros((n_s, n + n_p))])])
    E = np.concatenate([S_a @ inv_M @ (r + V), fiber_strain(gauges) - Hs @ y_prev])

    # update, eqs. (27)-(31)
    S_k = Ha_k @ P_minus @ Ha_k.T + R             # innovation covariance, eq. (27)
    e = d[k] - (Ha_k @ za_minus + E)              # innovation, eq. (28)
    G_k = np.linalg.solve(S_k.T, (P_minus @ Ha_k.T).T).T   # Kalman gain, eq. (29)
    za[k] = za_minus + G_k @ e                    # eq. (30)
    P = P_minus - G_k @ Ha_k @ P_minus            # eq. (31)
    P_diag[k] = np.diag(P)

    # FE-aided step (Sec. 3.2): impose y_hat on the FE model -> K_{k+1}, P_R
    K, P_R, ok = tangent_update(za[k, :n], K, P_R, dofs, V)
    if ok != 0:
        raise RuntimeError(f"tangent update failed at step {k}")
    MK_est[k] = section_response(sections)        # FE model is now at the estimated state
    EPS_est[k] = fiber_strain(gauges)
    if np.max(np.abs((np.diag(K) - np.diag(K_ref)) / np.diag(K_ref))) > K_UPDATE_TOL:
        K_ref = K.copy()
        H0inv, Aa, Qa, Ha = formulation(K_ref)
        n_updates += 1

ops.wipe()
print(f"Formulation rebuilt at {n_updates} of {n_steps - 1} steps (K_UPDATE_TOL = {K_UPDATE_TOL:g})")

# =============================================================================
# Results
# =============================================================================
t = np.arange(n_steps) * dt
ug_dd_true = ug_dd[:n_steps]


def nrmse(est, ref):
    return 100 * np.sqrt(np.mean((est - ref) ** 2)) / np.std(ref)


rows = [(dofs[20][0], "disp. 1st floor [m]", Y[:n_steps, dofs[20][0]]),
        (dofs[60][0], "disp. roof [m]", Y[:n_steps, dofs[60][0]]),
        (-1, "ground acc. [m/s²]", ug_dd_true)]
n_rows = len(rows) + n_s                          # one extra row per strain gauge
fig, ax = plt.subplots(n_rows, 2, figsize=(13, 2 * n_rows), gridspec_kw=dict(width_ratios=[3, 1]))
zoom = (t >= 6) & (t <= 9)
for i, (j, label, ref) in enumerate(rows):
    est, s = za[:, j], 2 * np.sqrt(P_diag[:, j])
    c = "C1" if j == -1 else "C0"
    for col, sl in enumerate([slice(None), zoom]):
        a = ax[i, col]
        a.fill_between(t[sl], (est - s)[sl], (est + s)[sl], color=c, alpha=0.25, lw=0, label="±2σ")
        a.plot(t[sl], ref[sl], "k", lw=1, label="true")
        a.plot(t[sl], est[sl], "--", color=c, lw=1, label="AEKF")
        a.grid(alpha=0.3)
    ax[i, 0].set_ylabel(label, fontsize=8)
    ax[i, 0].set_title(f"NRMSE = {nrmse(est, ref):.2f} %", fontsize=9, loc="right")
    print(f"{label:22s} NRMSE = {nrmse(est, ref):6.2f} %")
# strain gauges: measured (noisy), true, estimated (from the FE model at the estimate)
for g in range(n_s):
    i = len(rows) + g
    for col, sl in enumerate([slice(None), zoom]):
        a = ax[i, col]
        a.plot(t[sl], 1e6 * d[:n_steps][sl, len(sens) + g], color="0.75", lw=0.6, label="measured")
        a.plot(t[sl], 1e6 * EPS_true[:n_steps][sl, g], "k", lw=1, label="true")
        a.plot(t[sl], 1e6 * EPS_est[sl, g], "--", color="C0", lw=1, label="AEKF")
        a.grid(alpha=0.3)
    ax[i, 0].set_ylabel(f"strain ele {STRAIN_GAUGES[g][0]} [µε]", fontsize=8)
    ax[i, 0].set_title(f"NRMSE = {nrmse(EPS_est[:, g], EPS_true[:n_steps, g]):.2f} %", fontsize=9, loc="right")
    ax[i, 0].legend(fontsize=8, ncol=3, loc="upper right")
    print(f"{'strain ele ' + str(STRAIN_GAUGES[g][0]):22s} NRMSE = {nrmse(EPS_est[:, g], EPS_true[:n_steps, g]):6.2f} %")
ax[0, 0].legend(fontsize=8, ncol=3, loc="upper right")
ax[-1, 0].set_xlabel("time [s]")
ax[-1, 1].set_xlabel("time [s] (zoom)")
fig.suptitle(f"FE-aided AEKF, 5-storey 3-bay steel frame — {RECORD} ×{SCALE:g}, "
             f"acc. at nodes {SENSOR_NODES} ({NOISE:.0%} noise)"
             + (f" + strain gauges {[g[0] for g in STRAIN_GAUGES]} ({NOISE_STRAIN:.0%})" if n_s else ""),
             fontsize=10)
fig.tight_layout()
fig.savefig("figures/strainMeas_5storey_aekf_estimation.png", dpi=110)

# moment-curvature at selected sections: true vs estimated
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
for a, (i, name) in zip(ax.flat, enumerate(MK_SECTIONS)):
    a.plot(1e3 * MK_true[:n_steps, i, 0], 1e-3 * MK_true[:n_steps, i, 1], "k", lw=1, label="true")
    a.plot(1e3 * MK_est[:, i, 0], 1e-3 * MK_est[:, i, 1], "--", color="C3", lw=1, label="AEKF")
    a.set_title(name, fontsize=9)
    a.set_xlabel("curvature [10$^{-3}$ 1/m]")
    a.set_ylabel("moment [kNm]")
    a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8)
fig.suptitle("Moment-curvature: true (OpenSees) vs estimated (FE-aided AEKF)", fontsize=10)
fig.tight_layout()
fig.savefig("figures/strainMeas_5storey_moment_curvature.png", dpi=110)
plt.show()
