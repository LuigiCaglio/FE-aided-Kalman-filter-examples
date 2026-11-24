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
"""2-storey, 2-bay steel frame in OpenSeesPy (2-D, force-based fibre elements)."""

import openseespy.opensees as ops

from ops_helpers import i_section_fibers

GPa, MPa, mm = 1e9, 1e6, 1e-3

HEIGHT = 5.0     # total height [m] (two storeys of 2.5 m)
SPAN = 4.0       # bay width [m]
MASS = 5000.0    # nodal mass [kg]

# Wide-flange profiles: depth d, flange width bf, flange thickness tf, web thickness tw
PROFILES = {
    "IPE200": dict(d=200 * mm, bf=100 * mm, tf=8.5 * mm, tw=5.6 * mm),   # beams
    "HEB200": dict(d=200 * mm, bf=200 * mm, tf=15.0 * mm, tw=9.0 * mm),  # columns
}


def build_frame(E=210 * GPa, fy=180 * MPa):    # low fy: the frame yields under the scaled ground motion
    """Create the frame. Nodes: 1x = base, 2x = 1st floor, 3x = roof (x = column line 0,1,2)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # [horizontal, vertical, rotational]; horizontal = 0.8 x vertical; small rotational mass
    # so that M is invertible
    nodal_mass = [0.8 * MASS, MASS, 0.01]
    for level, y in [(10, 0.0), (20, HEIGHT / 2), (30, HEIGHT)]:
        for col in range(3):
            tag = level + col
            if level == 10:
                ops.node(tag, col * SPAN, y)
                ops.fix(tag, 1, 1, 1)
            else:
                ops.node(tag, col * SPAN, y, "-mass", *nodal_mass)

    # Steel02 (Giuffre-Menegotto-Pinto): hardening ratio b, transition parameters R0, cR1, cR2
    ops.uniaxialMaterial("Steel02", 1, fy, E, 0.02, 18.0, 0.925, 0.15)

    i_section_fibers(1, 1, **PROFILES["IPE200"])   # beams
    i_section_fibers(2, 1, **PROFILES["HEB200"])   # columns
    ops.beamIntegration("Lobatto", 1, 1, 5)
    ops.beamIntegration("Lobatto", 2, 2, 5)

    ops.geomTransf("Linear", 1)   # beams
    ops.geomTransf("PDelta", 2)   # columns

    for base in (10, 20):                                  # columns, storey by storey
        for col in range(3):
            i, j = base + col, base + 10 + col
            ops.element("forceBeamColumn", 100 * i + j, i, j, 2, 2)
    for floor in (20, 30):                                 # beams
        for bay in range(2):
            i = floor + bay
            ops.element("forceBeamColumn", 100 * i + i + 1, i, i + 1, 1, 1)



FLOOR_NODES = [20, 21, 22, 30, 31, 32]
BEAMS = [2021, 2122, 3031, 3132]


def gravity_loads(g=9.81):
    """Nodal gravity loads from the lumped vertical masses: {node: [Fx, Fy, Mz]}."""
    return {node: [0.0, -MASS * g, 0.0] for node in FLOOR_NODES}


def beam_gravity_w(g=9.81):
    """Uniform beam load [N/m] with the same total weight as the nodal option."""
    return 3 * MASS * g / (2 * SPAN)      # weight of one floor (3 nodes) over its 2 bays


def beam_equivalent_loads(g=9.81):
    """Equivalent nodal loads of the uniform beam load (fixed-end forces): {node: [Fx, Fy, Mz]}.

    Each beam end gets the shear -wL/2 and the moment -/+ wL^2/12 (end i / end j).
    """
    w, loads = beam_gravity_w(g), {}
    for ele in BEAMS:
        for node, sign in [(ele // 100, 1.0), (ele % 100, -1.0)]:
            fx, fy, mz = loads.get(node, [0.0, 0.0, 0.0])
            loads[node] = [fx, fy - w * SPAN / 2, mz - sign * w * SPAN**2 / 12]
    return loads


def add_gravity_pattern(kind="nodal", tag=999, g=9.81):
    """Gravity load pattern, to be applied in a static analysis and then held constant.

    kind = "nodal"       : point loads -m g at the floor nodes
    kind = "distributed" : uniform load on the beams (eleLoad -beamUniform), same total weight
    The masses stay lumped at the nodes in both cases.
    """
    ops.timeSeries("Linear", tag)
    ops.pattern("Plain", tag, tag)
    if kind == "nodal":
        for node, f in gravity_loads(g).items():
            ops.load(node, *f)
    elif kind == "distributed":
        for ele in BEAMS:
            ops.eleLoad("-ele", ele, "-type", "-beamUniform", -beam_gravity_w(g))
    else:
        raise ValueError(f"unknown gravity kind: {kind}")
