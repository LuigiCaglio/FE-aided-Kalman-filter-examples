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
"""OpenSeesPy helpers: ground-motion input, system matrices, damping, transient analysis,
section response, FE-aided tangent update."""

import numpy as np
import openseespy.opensees as ops


def read_at2(path):
    """Read a PEER-style .at2 record -> (acceleration [g], dt [s]).

    dt is taken from the 'DT=' entry of the 4th header line; the values follow the header.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[3].replace(",", " ").split()
    dt = float(header[header.index("DT=") + 1])
    acc = np.array(" ".join(lines[4:]).split(), dtype=float)
    return acc, dt


def i_section_fibers(sec_tag, mat_tag, d, bf, tf, tw, n_web=8, n_flange=2):
    """Fibre section of a doubly-symmetric I/H profile, bending about the local z axis.

    Three rectangles (bottom flange, web, top flange) meshed in layers along y:
    `n_flange` layers through each flange and `n_web` layers over the clear web.
    """
    hw = d / 2 - tf
    ops.section("Fiber", sec_tag)
    # patch('rect', matTag, nSubY, nSubZ, yI, zI, yJ, zJ)
    ops.patch("rect", mat_tag, n_flange, 1, -d / 2, -bf / 2, -hw, bf / 2)
    ops.patch("rect", mat_tag, n_web, 1, -hw, -tw / 2, hw, tw / 2)
    ops.patch("rect", mat_tag, n_flange, 1, hw, -bf / 2, d / 2, bf / 2)


def _setup_analysis():
    ops.wipeAnalysis()
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Transformation")
    ops.algorithm("Newton")


def dof_map():
    """{node: [eq. numbers of its 3 DOFs]} (-1 = fixed). Valid after an analysis has been set up."""
    return {n: ops.nodeDOFs(n) for n in ops.getNodeTags()}


def system_matrices():
    """Assembled M, C, K (free DOFs) and the DOF map, via the GimmeMCK integrator."""
    _setup_analysis()
    ops.algorithm("Linear")                       # GimmeMCK requires a linear algorithm
    ops.test("NormDispIncr", 1e-7, 100)
    ops.integrator("GimmeMCK", 1.0, 0.0, 0.0)
    ops.analysis("Transient")
    out = []
    for coeffs in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:   # M, C, K
        ops.integrator("GimmeMCK", *coeffs)
        ops.analyze(1, 0.0)
        n = ops.systemSize()
        out.append(np.array(ops.printA("-ret")).reshape(n, n).T)   # printA: column-major
    dofs = dof_map()
    ops.wipeAnalysis()
    M, C, K = out
    return M, C, K, dofs


def rayleigh_damping(zeta=0.05, modes=(1, 3)):
    """Rayleigh damping with ratio `zeta` on two modes; sets it in OpenSees and returns (a, b)."""
    w = np.sqrt(ops.eigen(max(modes)))
    wi, wj = w[modes[0] - 1], w[modes[1] - 1]
    a, b = np.linalg.solve([[1 / wi, wi], [1 / wj, wj]], [2 * zeta, 2 * zeta])
    ops.rayleigh(a, 0.0, 0.0, b)     # mass- and committed-stiffness-proportional
    return a, b


def nodal_vector(dofs, getter):
    """Collect a nodal quantity (ops.nodeDisp / nodeVel / nodeAccel) into a free-DOF vector."""
    n = 1 + max(max(d) for d in dofs.values())
    v = np.zeros(n)
    for node, eqs in dofs.items():
        vals = getter(node)
        for j, eq in enumerate(eqs):
            if eq >= 0:
                v[eq] = vals[j]
    return v


def section_response(sections):
    """Curvature and bending moment at the given (element, integration point) pairs.

    Integration points are numbered from 1. For a 2-D fibre section the section
    deformations are [axial strain, curvature] and the forces are [N, M].
    Returns an array of shape (len(sections), 2): [curvature, moment].
    """
    out = []
    for ele, ip in sections:
        kappa = ops.eleResponse(ele, "section", ip, "deformation")[1]
        moment = ops.eleResponse(ele, "section", ip, "force")[1]
        out.append([kappa, moment])
    return np.array(out).reshape(len(sections), 2)


# -----------------------------------------------------------------------------------------
# Strain gauges (Appendix B of [2]): a gauge is (element, integration point, y of the fibre)
# -----------------------------------------------------------------------------------------
def fiber_y(ele, ip, side):
    """Local y coordinate of the outer fibre ("bottom" = min y, "top" = max y) of a section."""
    y = np.array(ops.eleResponse(ele, "section", ip, "fiberData")).reshape(-1, 5)[:, 0]
    return y.min() if side == "bottom" else y.max()


def fiber_strain(gauges):
    """Strain at the gauges: eps = eps_0 - y * kappa (plane sections, eq. (B.4))."""
    out = []
    for ele, ip, y in gauges:
        eps0, kappa = ops.eleResponse(ele, "section", ip, "deformation")[:2]
        out.append(eps0 - y * kappa)
    return np.array(out)


def strain_row(ele, ip, y, dofs, n):
    """Displacement-to-strain row H^s = T_ms f_s T_sb k_b T_bg S_d^s, eq. (B.3), for a 2-D
    force-based element with linear (or P-Delta) geometric transformation."""
    (x1, y1), (x2, y2) = ops.nodeCoord(ops.eleNodes(ele)[0]), ops.nodeCoord(ops.eleNodes(ele)[1])
    L, b = np.hypot(x2 - x1, y2 - y1), np.arctan2(y2 - y1, x2 - x1)
    c, s = np.cos(b), np.sin(b)
    T_bg = np.array([[-c, -s, 0.0, c, s, 0.0],                          # eq. (B.6)
                     [-s / L, c / L, 1.0, s / L, -c / L, 0.0],
                     [-s / L, c / L, 0.0, s / L, -c / L, 1.0]])
    x = ops.sectionLocation(ele, ip)                                    # position along the element
    T_sb = np.array([[1.0, 0.0, 0.0], [0.0, x / L - 1.0, x / L]])       # eq. (B.5)
    f_s = np.array(ops.sectionFlexibility(ele, ip)).reshape(2, 2)       # section flexibility
    k_b = np.array(ops.eleResponse(ele, "basicStiffness")).reshape(3, 3)
    T_ms = np.array([1.0, -y])                                          # eq. (B.4)
    S_d = np.zeros((6, n))                                              # element DOFs -> global
    for a, node in enumerate(ops.eleNodes(ele)):
        for comp, eq in enumerate(dofs[node]):
            if eq >= 0:
                S_d[3 * a + comp, eq] = 1.0
    return T_ms @ f_s @ T_sb @ k_b @ T_bg @ S_d


def run_transient(dt, n_steps, dofs, ts_tag=1, sections=(), gauges=()):
    """Newmark (average acceleration) transient analysis.

    Returns displacement y, velocity y_dot and relative acceleration y_ddot of all free
    DOFs (n_steps+1 x n_dof, first row = initial state), the ground acceleration applied
    at each step (from the time series `ts_tag`), the curvature/moment history at
    `sections` (n_steps+1 x n_sections x 2) and the strain history at `gauges`
    (n_steps+1 x n_gauges).
    """
    _setup_analysis()
    ops.test("NormUnbalance", 1e-8, 20)
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    def state():
        return [nodal_vector(dofs, g) for g in (ops.nodeDisp, ops.nodeVel, ops.nodeAccel)] \
            + [ops.getLoadFactor(ts_tag), section_response(sections), fiber_strain(gauges)]

    hist = [state()]
    for k in range(n_steps):
        if ops.analyze(1, dt) != 0:
            raise RuntimeError(f"transient analysis failed at t = {ops.getTime():.3f} s")
        hist.append(state())
    Y, Yd, Ydd, ug_dd, MK, EPS = (np.array(x) for x in zip(*hist))
    return Y, Yd, Ydd, ug_dd, MK, EPS.reshape(len(hist), len(gauges))


def static_analysis(n_steps=10):
    """Apply the currently defined load patterns statically, then hold them constant."""
    _setup_analysis()
    ops.test("NormDispIncr", 1e-8, 100)
    ops.integrator("LoadControl", 1.0 / n_steps)
    ops.analysis("Static")
    ok = ops.analyze(n_steps)
    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()
    if ok != 0:
        raise RuntimeError("static (gravity) analysis failed")


def load_vector(loads, dofs):
    """Nodal loads {node: [Fx, Fy, Mz]} -> free-DOF vector."""
    return nodal_vector(dofs, lambda nd: loads.get(nd, [0.0, 0.0, 0.0]))


def tangent_update(y_hat, K, P_R, dofs, V=None, tag=1000):
    """FE-aided step of the AEKF (Sec. 3.2 of [2]): impose the estimated displacements.

    The static force increment  Delta f = K (y_hat - y_FE)  is applied to the FE model in
    one Newton load step, which brings it to y_hat. The model already carries the constant
    known loads V implicitly (the gravity pattern stays applied; element loads are inside
    the element resisting force), so the extra nodal pattern holds only P_R - V + Delta f.
    Returns the new tangent stiffness K, the new restoring forces P_R and the OpenSees
    return code.
    """
    V = np.zeros_like(P_R) if V is None else V
    y_FE = nodal_vector(dofs, ops.nodeDisp)
    f = P_R - V + K @ (y_hat - y_FE)

    ops.remove("loadPattern", tag)
    ops.remove("timeSeries", tag)
    ops.timeSeries("Constant", tag)
    ops.pattern("Plain", tag, tag)
    for node, eqs in dofs.items():
        if max(eqs) >= 0:
            ops.load(node, *[f[e] if e >= 0 else 0.0 for e in eqs])

    _setup_analysis()
    ops.test("NormDispIncr", 1e-7, 100)
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ok = ops.analyze(1)

    n = len(f)
    K_t = np.array(ops.printA("-ret")).reshape(n, n).T   # printA: column-major
    ops.wipeAnalysis()
    return K_t, f + V, ok


def plot_structure(title="", gauges=(), sensor_nodes=()):
    """Plot the OpenSees model with opsvis (node and element tags), marking sensors."""
    import opsvis
    import matplotlib.pyplot as plt
    opsvis.plot_model(node_labels=1, element_labels=1, gauss_points=False)
    ax = plt.gca()
    for nd in sensor_nodes:
        x, y = ops.nodeCoord(nd)
        ax.plot(x, y, ">", ms=12, color="C1", zorder=5)
        ax.annotate("acc.", (x, y), textcoords="offset points", xytext=(10, -4), color="C1")
    for ele, ip, *_ in gauges:
        (x1, y1), (x2, y2) = ops.nodeCoord(ops.eleNodes(ele)[0]), ops.nodeCoord(ops.eleNodes(ele)[1])
        f = ops.sectionLocation(ele, ip) / np.hypot(x2 - x1, y2 - y1)
        ax.plot(x1 + f * (x2 - x1), y1 + f * (y2 - y1), "s", ms=9, color="C3", zorder=5)
        ax.annotate("strain", (x1 + f * (x2 - x1), y1 + f * (y2 - y1)), textcoords="offset points",
                    xytext=(6, 6), color="C3")
    ax.set_title(title)
    return ax
