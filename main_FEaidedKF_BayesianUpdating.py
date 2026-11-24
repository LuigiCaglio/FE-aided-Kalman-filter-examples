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
"""Bayesian model updating of the steel frame with the FE-aided AEKF, following [2].

Unknown parameters theta = [E, fy] (elastic modulus and yield strength of the steel),
unknown input (ground acceleration), measured: noisy absolute accelerations.

1. "True" response with the true parameters, and the noisy measurements.
2. Negative log-posterior phi(theta), eq. (34): one FE-aided AEKF run per evaluation,
   uniform prior within the bounds.
3. MAP estimate, eq. (36): Nelder-Mead minimisation of phi.
4. Laplace approximation, eqs. (37)-(38): posterior covariance = inverse Hessian of phi.
5. Response and input estimation with the MAP parameters.

Run:  python main_FEaidedKF_BayesianUpdating.py      -> figures/bayesian_updating_posterior.png, figures/bayesian_updating_estimation.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import openseespy.opensees as ops

from frame_2storey_2bay_2D import build_frame
from ops_helpers import plot_structure
from fe_aided_kf import true_response, measurements, run_aekf
from bayesian_updating import NegLogPosterior, map_estimate, hessian_fd, laplace, fmt

RECORD = "synthetic_gm.at2"
SCALE = 3.0
SENSOR_NODES = [22, 32]
NOISE = 0.01
GRAVITY = "distributed"
S_LOAD = 1e-2                   # random-walk covariance S (kept fixed)
Q_STATE = 1e-20
K_UPDATE_TOL = 5e-3             # same as main_FEaidedKF.py

THETA_TRUE = np.array([210e9, 180e6])            # [E, fy] used to generate the data
BOUNDS = np.array([[170e9, 250e9],               # uniform prior on E
                   [130e6, 250e6]])              # uniform prior on fy
THETA_START = BOUNDS.mean(axis=1)                # optimiser start: centre of the prior box
NAMES, UNITS, SCALES = ["E", "f_y"], ["GPa", "MPa"], [1e9, 1e6]


# =============================================================================
# 1. True response and measurements
# =============================================================================
build_frame(*THETA_TRUE)                          # only to plot the model
plot_structure("2-storey 2-bay steel frame (opsvis)", sensor_nodes=SENSOR_NODES)
ops.wipe()

true = true_response(RECORD, SCALE, *THETA_TRUE, gravity=GRAVITY)
d, R = measurements(true, SENSOR_NODES, NOISE)
dt = true["dt"]

# =============================================================================
# 2. Negative log-posterior phi(theta), eq. (34)
# =============================================================================
def nll(theta):
    """-ln p(d_1:N | theta) from one FE-aided AEKF run, eq. (33)."""
    return run_aekf(d, R, dt, SENSOR_NODES, *theta, gravity=GRAVITY, s_load=S_LOAD,
                    q_state=Q_STATE, k_update_tol=K_UPDATE_TOL, progress=False)["nll"]


phi = NegLogPosterior(nll, theta_ref=THETA_START, bounds=BOUNDS)   # theta_ref: scaling only

# =============================================================================
# 3. MAP estimate, eq. (36)
# =============================================================================
print("MAP estimate (Nelder-Mead):")
t0 = time.time()
theta_map, _ = map_estimate(phi, THETA_START)
n_map = len(phi.history)

# =============================================================================
# 4. Laplace approximation, eqs. (37)-(38)
# =============================================================================
print("Hessian at the MAP (finite differences):")
H = hessian_fd(phi, theta_map, h=0.01)
cov = laplace(H)
std = np.sqrt(np.diag(cov))

print(f"\n{'':4s}{'true':>10s}{'MAP':>10s}{'post. std':>11s}{'CoV':>8s}")
for i in range(2):
    s = SCALES[i]
    print(f"{NAMES[i]:4s}{THETA_TRUE[i]/s:10.2f}{theta_map[i]/s:10.2f}"
          f"{std[i]/s:11.3f}{100*std[i]/theta_map[i]:7.2f}%   [{UNITS[i]}]")
print(f"phi evaluations: {n_map} (MAP) + {len(phi.history) - n_map} (Hessian) model runs, {time.time() - t0:.0f} s")

# =============================================================================
# 5. Estimation with the MAP parameters
# =============================================================================
est = run_aekf(d, R, dt, SENSOR_NODES, *theta_map, gravity=GRAVITY, s_load=S_LOAD,
               q_state=Q_STATE, k_update_tol=K_UPDATE_TOL)

# =============================================================================
# Figures
# =============================================================================
INK, BLUE, ORANGE = "#0b0b0b", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e7e6e2", "axes.edgecolor": "#b5b3ad"})
hist_theta = np.array([h[0] for h in phi.history])
hist_phi = np.array([h[1] for h in phi.history])
sc = np.array(SCALES)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
# (a) optimisation path and Laplace 95 % region
a = ax[0]
p = hist_theta[:n_map] / sc
a.plot(p[:, 0], p[:, 1], "-", color="#d6d5d0", lw=0.8, zorder=1)
s = a.scatter(p[:, 0], p[:, 1], c=hist_phi[:n_map] - hist_phi.min(), cmap="viridis_r", s=18,
              norm=plt.matplotlib.colors.LogNorm(vmin=1e-1), zorder=2)
fig.colorbar(s, ax=a, label="φ − φ_min")
w, v = np.linalg.eigh(cov / np.outer(sc, sc))
ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
a.add_patch(Ellipse(theta_map / sc, *(2 * np.sqrt(5.991 * w[::-1])), angle=ang,
                    fill=False, ec=BLUE, lw=1.5, zorder=3))
a.plot(*(THETA_START / sc), "s", ms=7, mfc="white", mec=INK, label="optimiser start", zorder=4)
a.plot(*(theta_map / sc), "o", ms=7, color=BLUE, label="MAP (95 % Laplace region)", zorder=4)
a.plot(*(THETA_TRUE / sc), "x", ms=9, mew=2, color=ORANGE, label="true", zorder=5)
a.set_xlabel("E [GPa]"); a.set_ylabel("f_y [MPa]")
a.set_title("Nelder–Mead path of the MAP search", loc="left")
a.legend(frameon=False, fontsize=8)
# (b), (c) marginal posteriors (Laplace)
for i in range(2):
    a = ax[1 + i]
    m, sd = theta_map[i] / sc[i], std[i] / sc[i]
    x = np.linspace(m - 5 * sd, m + 5 * sd, 400)
    a.fill_between(x, np.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                   color=BLUE, alpha=0.25, lw=0)
    a.axvline(m, color=BLUE, lw=1.5, label=f"MAP {m:.2f} ± {sd:.2f}")
    a.axvline(THETA_TRUE[i] / sc[i], color=ORANGE, lw=1.5, ls="--", label=f"true {THETA_TRUE[i]/sc[i]:.0f}")
    a.set_xlabel(f"{NAMES[i]} [{UNITS[i]}]"); a.set_yticks([])
    a.set_title(f"posterior of {NAMES[i]} (Laplace)", loc="left")
    a.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/bayesian_updating_posterior.png", dpi=140)

# estimation with the MAP parameters
t = np.arange(d.shape[0] - 1) * dt
dofs = true["dofs"]
rows = [("disp. roof [mm]", 1e3 * true["Y"][:-1, dofs[30][0]], lambda r: 1e3 * r["za"][:, dofs[30][0]]),
        ("ground acc. [m/s²]", true["ug_dd"][:-1], lambda r: r["za"][:, -1])]
fig, ax = plt.subplots(2, 2, figsize=(13, 6), gridspec_kw=dict(width_ratios=[3, 1.3]))
zoom = (t >= 6) & (t <= 9)
for i, (lab, ref, get) in enumerate(rows):
    for j, sl in enumerate([slice(None), zoom]):
        a = ax[i, j]
        a.plot(t[sl], ref[sl], color=INK, lw=1.3, label="true")
        e = get(est)
        nr = 100 * np.sqrt(np.mean((e - ref) ** 2)) / np.std(ref)
        a.plot(t[sl], e[sl], "--", color=BLUE, lw=1.0, label=f"AEKF at MAP (NRMSE {nr:.2f} %)")
    ax[i, 0].set_ylabel(lab)
ax[0, 0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax[1, 0].set_xlabel("time [s]"); ax[1, 1].set_xlabel("time [s] (zoom)")
fig.suptitle(f"Estimation with the MAP parameters {fmt(theta_map / sc)} [GPa, MPa]",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/bayesian_updating_estimation.png", dpi=140)
plt.show()
