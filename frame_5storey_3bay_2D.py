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
"""5-storey, 3-bay steel moment frame in OpenSeesPy (2-D), the structure of [1] and [2].

Force-based fibre elements (5 Lobatto points), Steel02 material (beams and columns as two
materials), P-Delta columns. The masses come from the self-weight and the floor loads
(DL + 0.3 LL over a 6 m tributary width), lumped at the element nodes; gravity loads are the
lumped vertical masses times g.

Node tags 'ij': i = level (1 = ground, 2 = 1st floor, ..., 6 = roof), j = column line 0-3.
Element tags = node tags of its two nodes, e.g. 1020 = ground-floor column on line 0.
"""

import numpy as np
import openseespy.opensees as ops

from ops_helpers import i_section_fibers

GPa, MPa, mm, kN = 1e9, 1e6, 1e-3, 1e3
G = 9.81

N_STOREY, N_BAY = 5, 3
H_GROUND, H_STOREY = 4.5, 3.5      # storey heights [m]
L_BAY = 6.0                        # bay width [m]
W_TRIB = 6.0                       # out-of-plane tributary width of the floors [m]
RHO = 7850.0                       # steel density [kg/m3]
DL, LL = 4.0 * kN, 2.0 * kN        # floor loads [N/m2]
DL_TOP, LL_TOP = 3.5 * kN, 1.5 * kN

# Profiles: depth d, flange width bf, flange thickness tf, web thickness tw, area A
PROFILES = {
    "IPE330": dict(d=330 * mm, bf=160 * mm, tf=11.5 * mm, tw=7.5 * mm, A=6.26062e-3),
    "IPE360": dict(d=360 * mm, bf=170 * mm, tf=12.7 * mm, tw=8.0 * mm, A=7.27292e-3),
    "IPE400": dict(d=400 * mm, bf=180 * mm, tf=13.5 * mm, tw=8.6 * mm, A=8.44636e-3),
    "HEB360": dict(d=360 * mm, bf=300 * mm, tf=22.5 * mm, tw=12.5 * mm, A=18.06328e-3),
    "HEB400": dict(d=400 * mm, bf=300 * mm, tf=24.0 * mm, tw=13.5 * mm, A=19.77778e-3),
    "HEB450": dict(d=450 * mm, bf=300 * mm, tf=26.0 * mm, tw=14.0 * mm, A=21.79778e-3),
}
BEAMS_PER_STOREY = ["IPE400", "IPE400", "IPE400", "IPE360", "IPE330"]
COLUMNS_PER_STOREY = ["HEB450", "HEB450", "HEB400", "HEB400", "HEB360"]


def level_height(i):
    """Elevation of level i (1 = ground)."""
    return 0.0 if i == 1 else H_GROUND + (i - 2) * H_STOREY


def _lumped_element_mass(n1, n2, q):
    """Add the mass of an element with mass per length q [kg/m] to its two end nodes."""
    (x1, y1), (x2, y2) = ops.nodeCoord(n1), ops.nodeCoord(n2)
    L = np.hypot(x2 - x1, y2 - y1)
    for nd in (n1, n2):
        m = list(ops.nodeMass(nd))
        # translational: half the element mass; rotational: small (element mass * L^2 / 400),
        # so that M is invertible
        mt, mr = 0.5 * q * L, q * L * L**2 / 400
        ops.mass(nd, m[0] + mt, m[1] + mt, m[2] + mr)


def build_frame(E=210 * GPa, fy=235 * MPa, E_col=None, fy_col=None):
    """Create the frame. E, fy: beams (and columns, unless E_col / fy_col are given)."""
    E_col = E if E_col is None else E_col
    fy_col = fy if fy_col is None else fy_col
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    for i in range(1, N_STOREY + 2):
        for j in range(N_BAY + 1):
            ops.node(10 * i + j, j * L_BAY, level_height(i))
            ops.mass(10 * i + j, 0.0, 0.0, 0.0)
    for j in range(N_BAY + 1):
        ops.fix(10 + j, 1, 1, 1)

    # Steel02: hardening ratio 0.015, R0 = 40, cR1 = 0.925, cR2 = 0.15
    ops.uniaxialMaterial("Steel02", 1, fy, E, 0.015, 40.0, 0.925, 0.15)            # beams
    ops.uniaxialMaterial("Steel02", 2, fy_col, E_col, 0.015, 40.0, 0.925, 0.15)    # columns
    tags = {}
    for k, name in enumerate(PROFILES, start=1):
        p = PROFILES[name]
        i_section_fibers(k, 1 if name.startswith("IPE") else 2, p["d"], p["bf"], p["tf"], p["tw"])
        ops.beamIntegration("Lobatto", k, k, 5)
        tags[name] = k
    ops.geomTransf("Linear", 1)     # beams
    ops.geomTransf("PDelta", 2)     # columns

    floor_mass = (DL + 0.3 * LL) * W_TRIB / G       # [kg/m]
    floor_mass_top = (DL_TOP + 0.3 * LL_TOP) * W_TRIB / G
    for s in range(1, N_STOREY + 1):                # storey s: levels s -> s+1
        col, beam = COLUMNS_PER_STOREY[s - 1], BEAMS_PER_STOREY[s - 1]
        for j in range(N_BAY + 1):
            n1, n2 = 10 * s + j, 10 * (s + 1) + j
            ops.element("forceBeamColumn", 100 * n1 + n2, n1, n2, 2, tags[col])
            _lumped_element_mass(n1, n2, RHO * PROFILES[col]["A"])
        q_floor = floor_mass_top if s == N_STOREY else floor_mass
        for j in range(N_BAY):
            n1, n2 = 10 * (s + 1) + j, 10 * (s + 1) + j + 1
            ops.element("forceBeamColumn", 100 * n1 + n2, n1, n2, 1, tags[beam])
            _lumped_element_mass(n1, n2, RHO * PROFILES[beam]["A"] + q_floor)


def gravity_loads(g=G):
    """Nodal gravity loads from the lumped vertical masses: {node: [Fx, Fy, Mz]}."""
    return {nd: [0.0, -ops.nodeMass(nd)[1] * g, 0.0] for nd in ops.getNodeTags() if nd >= 20}


def add_gravity_pattern(kind="nodal", tag=999, g=G):
    """Gravity load pattern (nodal m*g), applied in a static analysis and then held constant."""
    if kind != "nodal":
        raise ValueError("the 5-storey frame uses nodal gravity loads (lumped masses * g)")
    ops.timeSeries("Linear", tag)
    ops.pattern("Plain", tag, tag)
    for node, f in gravity_loads(g).items():
        ops.load(node, *f)
