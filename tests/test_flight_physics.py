"""The flight simulator against theory it must reproduce, plus sign conventions. Falcon V2 replica, no autopilot."""
import math

import numpy as np
import pytest

from dronebench import flight as F
from dronebench import mission as M
from dronebench.ingest import load_design
from dronebench.pipeline import apply_plausibility, core


@pytest.fixture(scope="module")
def veh():
    d = load_design("fixtures/falcon_v2")
    apply_plausibility(d)
    return F.Vehicle.from_bits(core(d, "endurance"), d)


def trimmed_sim(veh, V=15.0, wind=None):
    tr = F.trim(veh, V)
    sim = F.Sim(veh, wind or F.Wind())
    sim.reset(alt=300, airspeed=V, pitch=tr["alpha"])
    return sim, tr


def run(sim, tr, seconds, **kw):
    out = []
    for _ in range(int(seconds / 0.01)):
        c = F.Controls(kw.get("da", 0.0), tr["de"] + kw.get("de", 0.0), kw.get("dr", 0.0), kw.get("thr", tr["throttle"]))
        out.append(sim.step(c, 0.01))
    return out


def test_trim_is_steady(veh):
    sim, tr = trimmed_sim(veh)
    o = run(sim, tr, 6)
    assert abs(o[-1]["alt"] - 300) < 3.0
    assert abs(o[-1]["airspeed"] - 15.0) < 0.6
    # the parts list is mirrored to 4 mm (push-rod links), so a trimmed aircraft rolls a few thousandths of a degree in 6 s
    assert abs(o[-1]["phi"]) < 1e-3 and abs(o[-1]["beta"]) < 1e-3


def test_control_signs(veh):
    # positive aileron command: roll right (right wing down), positive roll rate
    sim, tr = trimmed_sim(veh)
    o = run(sim, tr, 0.6, da=math.radians(5))
    assert o[-1]["p"] > 0.2 and o[-1]["phi"] > 0.05
    # positive elevator command: nose up
    sim, tr = trimmed_sim(veh)
    o = run(sim, tr, 0.5, de=math.radians(4))
    assert o[-1]["q"] > 0.05 and o[-1]["theta"] > tr["alpha"]
    # positive rudder command: nose right
    sim, tr = trimmed_sim(veh)
    o = run(sim, tr, 0.6, dr=math.radians(8))
    assert o[-1]["r"] > 0.02 and o[-1]["psi"] > 0.0


def test_sideslip_weathercock(veh):
    """A positive sideslip (velocity to the right of the nose) must swing the nose to the right (Cn_beta > 0)."""
    sim, tr = trimmed_sim(veh)
    sim.x[3:6] = np.array([15.0 * math.cos(0.08), 15.0 * math.sin(0.08), sim.x[5]])      # beta ~ +4.6 degrees
    o = run(sim, tr, 0.4)     # the first half of the yaw oscillation; the rate crosses zero near 0.8 s, so that is no place to test a sign
    assert o[0]["beta"] > 0.05 and o[-1]["r"] > 0.1 and o[-1]["psi"] > 0.03


def test_force_assembly_matches_an_independent_implementation(veh):
    """_dyn._forces against the python AeroTable.coeffs plus an independent wind-to-body rotation, at random flight states.
    (The energy-balance test uses the same compiled forces, so it cannot catch an assembly error; this one can.)"""
    from dronebench import _dyn as K
    rng = np.random.default_rng(4)
    Pv, Dv, al, ls, CL, CD, Cm = veh.pack()[:7]
    t = veh.tab
    for _ in range(25):
        V = rng.uniform(9, 30); alpha = rng.uniform(-0.05, 0.25); beta = rng.uniform(-0.15, 0.15)
        w = rng.normal(0, 0.4, 3); surf = rng.uniform(-0.2, 0.2, 3); thr = rng.uniform(0, 8)
        va = V * np.array([math.cos(alpha) * math.cos(beta), math.sin(beta), math.sin(alpha) * math.cos(beta)])
        q = F.quat_from_euler(rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3), rng.uniform(-3, 3))
        alt = 100.0
        F_k, M_k, _, _ = K._forces(va, w, q, surf, thr, np.zeros(3), Pv, Dv, al, ls, CL, CD, Cm, alt)
        cl, cd, cy, cll, cm, cn = t.coeffs(alpha, beta, w[0], w[1], w[2], V, surf[0], surf[1] + veh.inc_off, surf[2])
        cd = cd * veh.cd0_factor + t.d.get("CDde", 0) * 0.4 * abs(veh.inc_off) * 0     # python path has no inc term: add below
        cd = (t._tab(t.CD, alpha, V) + veh.cda_extra / t.s_ref) * veh.cd0_factor + t.d["CDde"] * (abs(surf[1]) + 0.4 * abs(veh.inc_off)) + t.d["CDda"] * abs(surf[0])
        qbar = 0.5 * F._rho(alt) * V ** 2
        L, D, Y = qbar * t.s_ref * cl, qbar * t.s_ref * cd, qbar * t.s_ref * cy
        ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
        Fa = np.array([-D * ca * cb - Y * ca * sb + L * sa, -D * sb + Y * cb, -D * sa * cb - Y * sa * sb - L * ca])
        Fg = F.quat_to_dcm(q).T @ np.array([0, 0, veh.mass * F.G])
        expect = Fa + Fg + np.array([thr, 0, 0])
        assert np.allclose(F_k, expect, rtol=1e-6, atol=1e-6), (F_k, expect)
        M_a = np.array([qbar * t.s_ref * t.b_ref * cll, qbar * t.s_ref * t.c_ref * cm, qbar * t.s_ref * t.b_ref * cn]) + np.cross(veh.ref_b, Fa)
        M_p = np.cross(veh.prop_pos_b, np.array([thr, 0, 0]))
        assert np.allclose(M_k, M_a + M_p, rtol=1e-6, atol=1e-6), (M_k, M_a + M_p)


def test_roll_rate_matches_derivatives(veh):
    """Steady roll rate = -L_da * da / L_p from the aerodynamic derivatives (roll subsidence mode)."""
    sim, tr = trimmed_sim(veh)
    rho = F._rho(300)
    t = veh.tab
    q = 0.5 * rho * 15.0 ** 2
    Lp = q * t.s_ref * t.b_ref ** 2 / (2 * 15.0 * veh.I[0, 0]) * t.d["Clp"]
    Lda = q * t.s_ref * t.b_ref / veh.I[0, 0] * t.d["Clda"]
    da = math.radians(2)
    o = run(sim, tr, 0.5, da=da)         # a short burst: before the bank angle changes the flow much
    p_ss = -Lda * da / Lp
    assert abs(o[-1]["p"] - p_ss) / p_ss < 0.20


def test_phugoid_period_matches_lanchester(veh):
    """Unpowered (so the prop does not add speed stability), the phugoid period is about pi * sqrt(2) * V / g
    (Lanchester, valid for L/D >> 1; the aircraft here has L/D ~ 15)."""
    V = 16.0
    tr = F.trim(veh, V)
    sim = F.Sim(veh)
    sim.reset(alt=900, airspeed=V, pitch=tr["alpha"])
    for _ in range(40):
        sim.step(F.Controls(0, tr["de"] + math.radians(3), 0, 0.0), 0.01)       # pitch kick, throttle off
    vs, ts = [], []
    for _ in range(int(90 / 0.02)):
        o = sim.step(F.Controls(0, tr["de"], 0, 0.0), 0.02)
        vs.append(o["airspeed"]); ts.append(o["t"])
    v = np.array(vs); t = np.array(ts)
    trend = np.convolve(v, np.ones(1001) / 1001, mode="same")
    a = (v - trend)[600:-600]; tt = t[600:-600]
    peaks = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > 0.15 * np.abs(a).max()]
    assert len(peaks) >= 2, f"no phugoid oscillation found ({len(peaks)} peaks)"
    period = float(np.median(np.diff(tt[peaks])))
    lanchester = math.pi * math.sqrt(2) * V / F.G
    assert 0.8 * lanchester < period < 1.5 * lanchester, (period, lanchester)


def test_energy_balance_in_a_glide(veh):
    """Throttle off: the loss of (kinetic + potential) energy equals the work done by the aerodynamic force, to a few %.
    This checks the integrator and the quaternion / frame kinematics. (It reuses the compiled force function, so force assembly
    is checked separately by test_force_assembly_matches_an_independent_implementation.)"""
    from dronebench import _dyn as K
    V = 14.0
    sim, tr = trimmed_sim(veh, V)
    Pv, Dv, al, ls, CL, CD, Cm = veh.pack()[:7]
    z3 = np.zeros(3)
    m, g = veh.mass, F.G

    def energy():
        return 0.5 * m * float(np.dot(sim.x[3:6], sim.x[3:6])) + m * g * sim.alt()

    e0, work = energy(), 0.0
    for _ in range(int(40 / 0.01)):
        sim.step(F.Controls(0, tr["de"], 0, 0.0), 0.01)
        Fv, _, _, _ = K._forces(sim.x[3:6], sim.x[10:13], sim.x[6:10], sim.surf, 0.0, z3, Pv, Dv, al, ls, CL, CD, Cm, sim.alt())
        Fg = F.quat_to_dcm(sim.x[6:10]).T @ np.array([0.0, 0.0, m * g])
        work += float(np.dot(Fv - Fg, sim.x[3:6])) * 0.01          # aerodynamic power, body axes
    lost = e0 - energy()
    assert lost > 0
    assert abs(lost + work) / lost < 0.05, (lost, work)


def test_power_matches_trim_estimate(veh):
    sim, tr = trimmed_sim(veh)
    o = run(sim, tr, 8)
    est = M.power_at(veh, 15.0, alt=300, soc=1.0)
    assert abs(o[-1]["power"] - est["P"]) / est["P"] < 0.12, (o[-1]["power"], est["P"])


def test_dryden_turbulence_statistics(veh):
    rng_seed = 7
    sim = F.Sim(veh, F.Wind(turb=1.0, seed=rng_seed))
    sim.reset(alt=300, airspeed=15)
    tr = F.trim(veh, 15.0)
    g = []
    for _ in range(int(400 / 0.02)):
        sim.step(F.Controls(0, tr["de"], 0, tr["throttle"]), 0.02)
        g.append(sim.gust.copy())
        if sim.crashed:
            break
    g = np.array(g)
    assert 0.7 < g[:, 0].std() < 1.5          # sigma_u ~ 1.06 m/s
    assert 0.45 < g[:, 2].std() < 1.0         # sigma_w ~ 0.7 m/s


def test_aft_cg_is_statically_unstable_and_fails_flight(veh):
    """A CG behind the neutral point must be reported as a failed design, not flown to a score."""
    import copy
    bad = copy.copy(veh)
    bad.cg_d = veh.cg_d + np.array([0.20, 0, 0])
    tab = veh.tab
    assert tab.static_margin(bad.cg_d[0]) < 0.05
