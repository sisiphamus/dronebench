"""Missions flown in the simulator, and the flight metrics DroneBench reports from them.

What is flown (autopilot, wind and turbulence on) and what is solved from trim (cheap sweeps used to pick speeds):

    performance_sweep   trim at many airspeeds: power, stall speed, best-endurance and best-range speed, top speed,
                        best rate of climb. Solved, not flown.
    fly_climb           full-throttle climb at best-climb speed: measured rate of climb and energy.
    fly_orbit           circle at constant altitude in wind and turbulence: measured battery power.
    fly_legs            upwind and downwind straight legs: measured energy per ground kilometre.

Endurance is (usable battery energy) / (mean flown orbit power), at two states of charge so voltage sag counts.
Range is usable energy / flown energy per ground km (mean of an upwind and a downwind leg). Scored segments use calm air and a
steady wind (deterministic). Turbulence is flown for pass/fail only: its realisation is too noisy to score. Top speed, climb and stall come from the sweep and the climb test.
Nothing here is calibrated to a particular aircraft; `cd0_factor` (aero.py) is the one explicit tuning knob.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import autopilot as AP
from . import flight as Fl
from . import _dyn as K

DT = 0.01


@dataclass
class Condition:
    """Weather for one flown segment. Same conditions for every design, so scores are comparable."""
    wind_ms: float = 3.0
    wind_from_deg: float = 270.0          # compass direction the wind blows FROM (north = 0)
    turb: float = 1.0
    seed: int = 1
    name: str = "light"

    def wind(self) -> Fl.Wind:
        to = math.radians(self.wind_from_deg + 180.0)       # direction it blows toward
        return Fl.Wind(mean=(self.wind_ms * math.cos(to), self.wind_ms * math.sin(to), 0.0), turb=self.turb, seed=self.seed)


# Two kinds of weather, kept apart on purpose.
#   SCORED: calm air and a steady wind. Deterministic (no random turbulence), so a design's score is the same every time and
#           two designs differ only because the aircraft differs. Wind enters range and energy through the flown legs and orbits.
#   ROBUST: turbulence (random Dryden gusts). Flown for pass/fail only: can the aircraft still be flown, within load and stall
#           limits? A 30 to 130 s turbulence realisation moves average power by tens of percent through sustained up- and
#           down-drafts, so it must not decide the score. `seed_offset` rotates the seeds (hold some back to test overfitting).
SCORED = (Condition(0.0, 0, 0.0, 10, "calm"), Condition(5.0, 250, 0.0, 13, "wind"))
ROBUST = (Condition(3.0, 270, 1.0, 11, "light"), Condition(6.0, 60, 2.0, 12, "gusty"))
CONDITIONS = SCORED + ROBUST
SOC_FOR_ORBIT = (1.0, 0.5)          # state of charge of the two scored orbits: voltage sag counts

# ---------------------------------------------------------------- sweeps (trim)
def power_at(veh: Fl.Vehicle, V: float, alt: float = 100.0, soc: float = 0.8):
    """Trim at V: (battery power W, thrust N, throttle, alpha, de) or None if it cannot be flown (throttle > 1)."""
    tr = Fl.trim(veh, V, alt=alt)
    pt = veh.pt
    if not pt.trains:
        return None
    vms, vas, T, I = pt.tables()
    vterm = pt.cells * float(np.interp(soc, pt.curve[:, 0], pt.curve[:, 1]))
    vm = tr["throttle"] * 0.98 * vterm
    cur = float(K._bilin(I, vms, vas, vm, V)) * Fl._rho(alt) / Fl.RHO0
    p = pt.n_trains * vm * cur / pt.esc_eff
    return {"V": V, "P": p, "T": tr["thrust"], "throttle": tr["throttle"], "alpha": tr["alpha"], "de": tr["de"], "ok": tr["ok"]}


def stall_speed(veh: Fl.Vehicle, alt: float = 100.0) -> float:
    rho = Fl._rho(alt)
    return math.sqrt(2 * veh.mass * Fl.G / (rho * veh.tab.s_ref * max(veh.tab.cl_max, 0.3)))


def performance_sweep(veh: Fl.Vehicle, alt: float = 100.0) -> dict:
    vs = stall_speed(veh, alt)
    grid = np.linspace(max(1.3 * vs, 6.0), 36.0, 24)      # never plan to fly closer than 1.3 x stall
    rows = [power_at(veh, float(V), alt) for V in grid]
    rows = [r for r in rows if r is not None and r["ok"]]
    out = {"v_stall": vs, "rows": rows}
    if not rows:
        out.update(v_best_endurance=float("nan"), v_best_range=float("nan"), v_max=float("nan"), roc_max=float("nan"), v_best_climb=float("nan"))
        return out
    P = np.array([r["P"] for r in rows]); V = np.array([r["V"] for r in rows])
    out["v_best_endurance"] = float(V[np.argmin(P)]); out["p_min"] = float(P.min())
    out["v_best_range"] = float(V[np.argmin(P / V)])
    out["wh_per_km_best"] = float((P / V).min() / 3.6)
    # top speed: highest V where trim needs no more than full throttle (the sweep already drops V that cannot be flown)
    hi = float(V.max())
    if hi >= 35.9:
        # refine upward beyond the grid, but not past the 50 m/s edge of the propeller table
        for Vx in np.arange(36.0, 49.0, 2.0):
            r = power_at(veh, float(Vx), alt)
            if r is None or not r["ok"]:
                break
            hi = float(Vx)
    out["v_max"] = hi
    # rate of climb: excess thrust at full throttle, steady, ROC = (T_avail - T_req) V / W
    pt = veh.pt
    vterm = pt.cells * float(np.interp(0.8, pt.curve[:, 0], pt.curve[:, 1]))
    best, vbc = -1e9, float("nan")
    for r in rows:
        avail = pt.thrust_at(0.98 * vterm, r["V"], Fl._rho(alt))
        roc = (avail - r["T"]) * r["V"] / (veh.mass * Fl.G)
        if roc > best:
            best, vbc = roc, r["V"]
    out["roc_max"], out["v_best_climb"] = float(best), float(vbc)
    return out


# ---------------------------------------------------------------- flown segments
@dataclass
class Flown:
    ok: bool
    reason: str = ""
    seconds: float = 0.0
    wh: float = 0.0
    mean_power: float = float("nan")
    ground_km: float = 0.0
    max_n: float = 1.0
    stalled_s: float = 0.0
    alt_rms: float = 0.0
    speed_rms: float = 0.0
    track_rms: float = 0.0
    wh_per_km: float = float("nan")
    climb_ms: float = float("nan")
    max_bank_deg: float = 0.0
    notes: list = field(default_factory=list)


def _new_sim(veh, cond: Condition, alt, V, soc=1.0, heading=0.0, pos=(0.0, 0.0), alpha0=None):
    sim = Fl.Sim(veh, cond.wind(), dt=DT)
    sim.reset(alt=alt, airspeed=V, heading=heading, pos=pos, pitch=alpha0, soc=soc)
    return sim


def fly_orbit(veh: Fl.Vehicle, cond: Condition, V: float, soc: float = 1.0, duration: float = 130.0, radius: float = 150.0,
              alt: float = 120.0, settle: float = 25.0) -> Flown:
    ap = AP.Autopilot(veh, V, alt=alt)
    if not ap.ok_controls:
        return Flown(False, "control surfaces have no authority")
    sim = _new_sim(veh, cond, alt, V, soc, alpha0=ap.tr["alpha"], pos=(0.0, -radius))
    center = (0.0, 0.0)
    n_s, p_sum, alt_e, v_e, tr_e = 0, 0.0, 0.0, 0.0, 0.0
    wh0 = None
    bank = 0.0
    for i in range(int(duration / DT)):
        o = sim.obs()
        chi, phi_ff = AP.orbit_course(o, center, radius, +1)
        ctl = ap.update(o, AP.Command(course=chi, alt=alt, airspeed=V, phi_ff=phi_ff), DT)
        o = sim.step(ctl, DT)
        bank = max(bank, abs(o["phi"]))
        if sim.crashed:
            return Flown(False, "crashed", sim.t, max_n=sim.max_n)
        if sim.structural_fail:
            return Flown(False, f"structural failure at {sim.max_n:.1f} g (limit {veh.n_ult:.1f} g)", sim.t, max_n=sim.max_n)
        if sim.t >= settle:
            if wh0 is None:
                wh0 = sim.used_wh()
            n_s += 1; p_sum += o["power"]
            alt_e += (o["alt"] - alt) ** 2; v_e += (o["airspeed"] - V) ** 2
            tr_e += (math.hypot(o["pos"][0], o["pos"][1]) - radius) ** 2
        if abs(o["alt"] - alt) > 45 or o["airspeed"] < 0.9 * Fl_vs(veh):
            return Flown(False, "lost altitude or speed (cannot hold the orbit)", sim.t, max_n=sim.max_n, stalled_s=sim.stalled_time)
    n_s = max(n_s, 1)
    return Flown(True, "", sim.t, wh=sim.used_wh() - (wh0 or 0.0), mean_power=p_sum / n_s, max_n=sim.max_n, stalled_s=sim.stalled_time,
                 alt_rms=math.sqrt(alt_e / n_s), speed_rms=math.sqrt(v_e / n_s), track_rms=math.sqrt(tr_e / n_s),
                 max_bank_deg=math.degrees(bank))


def Fl_vs(veh):
    return stall_speed(veh)


def fly_legs(veh: Fl.Vehicle, cond: Condition, V: float, soc: float = 0.8, leg_s: float = 45.0, alt: float = 120.0) -> Flown:
    """Fly one leg with the wind and one against it (heading changes by 180 degrees); return Wh per ground km."""
    ap = AP.Autopilot(veh, V, alt=alt)
    if not ap.ok_controls:
        return Flown(False, "control surfaces have no authority")
    to = math.radians(cond.wind_from_deg + 180.0)
    tot_wh, tot_km, p_sum, n_s = 0.0, 0.0, 0.0, 0
    alt_e, track = 0.0, 0.0
    per_leg = []
    for sgn in (+1, -1):
        heading = to if sgn > 0 else to + math.pi
        sim = _new_sim(veh, cond, alt, V, soc, heading=heading, alpha0=ap.tr["alpha"])
        p0, p1 = (0.0, 0.0), (math.cos(heading) * 5000, math.sin(heading) * 5000)
        sim.pt_state[2] = (1 - soc) * veh.pt.cap_wh
        ap2 = AP.Autopilot(veh, V, alt=alt)
        wh0, x0, settle = None, None, 12.0
        for i in range(int(leg_s / DT)):
            o = sim.obs()
            chi, e_y, _ = AP.line_course(o, p0, p1)
            ctl = ap2.update(o, AP.Command(course=chi, alt=alt, airspeed=V), DT)
            o = sim.step(ctl, DT)
            if sim.crashed or sim.structural_fail:
                return Flown(False, "crashed" if sim.crashed else "structural failure", sim.t, max_n=sim.max_n)
            if sim.t >= settle:
                if wh0 is None:
                    wh0, x0 = sim.used_wh(), o["pos"][:2].copy()
                p_sum += o["power"]; n_s += 1; alt_e += (o["alt"] - alt) ** 2; track += e_y ** 2
        d = float(np.linalg.norm(sim.x[0:2] - x0)) / 1000.0
        tot_km += d
        tot_wh += sim.used_wh() - wh0
        per_leg.append((sim.used_wh() - wh0) / max(d, 1e-9))
    if tot_km <= 0:
        return Flown(False, "no progress over the ground")
    # equal-distance round trip: the mean of the per-leg Wh/km (equal-time legs would let a tailwind cancel a headwind)
    return Flown(True, "", leg_s * 2, wh=tot_wh, ground_km=tot_km, mean_power=p_sum / max(n_s, 1), max_n=sim.max_n,
                 alt_rms=math.sqrt(alt_e / max(n_s, 1)), track_rms=math.sqrt(track / max(n_s, 1)),
                 wh_per_km=float(np.mean(per_leg)))


def fly_climb(veh: Fl.Vehicle, cond: Condition, V: float, alt0: float = 80.0, alt1: float = 400.0, max_s: float = 60.0) -> Flown:
    """Full throttle, hold best-climb airspeed with pitch; the rate of climb is measured between 120 m and 300 m."""
    ap = AP.Autopilot(veh, V, alt=alt0)
    if not ap.ok_controls:
        return Flown(False, "control surfaces have no authority")
    sim = _new_sim(veh, cond, alt0, V, 1.0, alpha0=ap.tr["alpha"])
    t_a = t_b = None
    wh_a = 0.0
    for i in range(int(max_s / DT)):
        o = sim.obs()
        ctl = ap.update_climb(o, AP.Command(course=0.0, alt=alt1, airspeed=V), DT)
        o = sim.step(ctl, DT)
        if sim.crashed or sim.structural_fail:
            return Flown(False, "crashed" if sim.crashed else "structural failure", sim.t, max_n=sim.max_n)
        if t_a is None and o["alt"] >= 120.0:
            t_a, wh_a = sim.t, sim.used_wh()
        if t_a is not None and o["alt"] >= 300.0:
            t_b = sim.t
            break
    if t_a is None:
        return Flown(False, "could not climb 40 m in 60 s", sim.t, climb_ms=0.0, max_n=sim.max_n)
    t_end = t_b if t_b is not None else sim.t
    top = 300.0 if t_b is not None else sim.alt()
    roc = (top - 120.0) / max(t_end - t_a, 1e-6)
    return Flown(True, "", sim.t, wh=sim.used_wh() - wh_a, climb_ms=roc, max_n=sim.max_n)


# ---------------------------------------------------------------- parallel execution of independent segments
_CTX: dict = {}


def _run_task(t):
    kind, args = t
    veh = _CTX["veh"]
    fn = {"orbit": fly_orbit, "legs": fly_legs, "climb": fly_climb}[kind]
    try:
        return fn(veh, *args)
    except Exception as e:           # a segment that raises is a failed segment, never a failed evaluation
        return Flown(False, f"{type(e).__name__}: {e}")


def run_segments(veh: Fl.Vehicle, tasks: list) -> list:
    """Run flown segments in forked worker processes (the compiled simulator and the aero tables are shared)."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor
    _CTX["veh"] = veh
    n = min(len(tasks), max(1, (os.cpu_count() or 2) - 1), 8)
    if n <= 1:
        return [_run_task(t) for t in tasks]
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork")) as ex:
            return list(ex.map(_run_task, tasks, timeout=400))
    except Exception:      # a stuck or broken worker pool (fork with threads can deadlock): fall back to one process
        return [_run_task(t) for t in tasks]


# ---------------------------------------------------------------- the report
def evaluate_flight(veh: Fl.Vehicle, preset: str = "endurance", scored=SCORED, robust=ROBUST, payload_fraction: float = 0.0,
                    seed_offset: int = 0) -> dict:
    """Everything the flown evaluation says about one design. Never raises: failures become failed checks with a reason."""
    out = {"ok": False, "fidelity": "flown 6-DOF (AeroSandbox aero, numba RK4)", "scored_conditions": [c.name for c in scored],
           "robust_conditions": [c.name for c in robust], "checks": [], "notes": []}
    chk = out["checks"]

    def add(name, value, limit, ok, detail=""):
        chk.append({"name": name, "value": value, "limit": limit, "pass": bool(ok), "detail": detail})

    try:
        sw = performance_sweep(veh)
    except Exception as e:      # a design with no trim must still produce a report
        out["reason"] = f"trim failed: {type(e).__name__}: {e}"
        add("trim", "none", "trimmable", False, out["reason"])
        return out
    out["sweep"] = {k: v for k, v in sw.items() if k != "rows"}
    out["usable_wh"] = veh.pt.usable * veh.pt.cap_wh
    tab = veh.tab
    sm = tab.static_margin(float(veh.cg_d[0]))
    out["static_margin"] = sm
    out["cl_max"] = tab.cl_max
    out["tail_inc_deg"] = math.degrees(veh.inc_off)
    if not sw["rows"]:
        out["reason"] = "no airspeed can be trimmed with the available thrust"
        add("trim", "none", "trimmable", False, out["reason"])
        return out
    v_e = max(sw["v_best_endurance"], 1.45 * sw["v_stall"])      # loiter with margin for bank and gusts
    v_r = max(sw["v_best_range"], 1.4 * sw["v_stall"])
    cruise = veh.v_design
    rb = [Condition(c.wind_ms, c.wind_from_deg, c.turb, c.seed + seed_offset, c.name) for c in robust]
    # ---- flown segments (independent, so flown in parallel)
    tasks = [("orbit", (c, v_e, SOC_FOR_ORBIT[k % len(SOC_FOR_ORBIT)])) for k, c in enumerate(scored)]
    tasks += [("legs", (c, v_r)) for c in scored]
    tasks += [("climb", (scored[0], sw["v_best_climb"]))]
    tasks += [("orbit", (c, v_e, 0.8)) for c in rb]
    tasks += [("legs", (rb[-1], v_r))]
    res = run_segments(veh, tasks)
    ns, nr = len(scored), len(rb)
    orbit_s, legs_s, climb = res[:ns], res[ns:2 * ns], res[2 * ns]
    orbit_r, legs_r = res[2 * ns + 1:2 * ns + 1 + nr], res[2 * ns + 1 + nr]
    out["orbit"] = [r.__dict__ for r in orbit_s]
    out["legs"] = [r.__dict__ for r in legs_s]
    out["climb"] = climb.__dict__
    out["robust"] = {"orbit": [r.__dict__ for r in orbit_r], "legs": legs_r.__dict__}
    # ---- checks (same list for every design; each has a value, a limit and a reason when it fails)
    add("static_margin", round(sm, 3), [0.05, 0.25], 0.05 <= sm <= 0.25, "vortex-lattice neutral point, fuselage-corrected")
    add("stall_margin", round(cruise / sw["v_stall"], 2), [1.3, None], cruise >= 1.3 * sw["v_stall"] and cruise > 0,
        f"cruise {cruise:.1f} m/s vs stall {sw['v_stall']:.1f} m/s")
    add("climb", round(climb.climb_ms if climb.climb_ms == climb.climb_ms else 0.0, 2), [2.0, None],
        climb.ok and climb.climb_ms >= 2.0, climb.reason or "measured in a full-throttle climb")
    for r, c in zip(orbit_s, scored):
        add(f"orbit_{c.name}", "flown" if r.ok else "failed", "completes", r.ok,
            r.reason or f"alt rms {r.alt_rms:.1f} m, speed rms {r.speed_rms:.2f} m/s, bank {r.max_bank_deg:.0f} deg")
    for r, c in zip(legs_s, scored):
        add(f"legs_{c.name}", "flown" if r.ok else "failed", "completes", r.ok, r.reason)
    for r, c in zip(orbit_r, rb):
        ok = r.ok and r.alt_rms <= 12.0 and r.speed_rms <= 4.0
        add(f"gusts_{c.name}", "flown" if ok else "failed", "completes, alt rms <= 12 m, speed rms <= 4 m/s", ok,
            r.reason or f"turbulence {c.turb:g}, wind {c.wind_ms:g} m/s: alt rms {r.alt_rms:.1f} m, speed rms {r.speed_rms:.2f} m/s, bank {r.max_bank_deg:.0f} deg")
    add("gusts_legs", "flown" if legs_r.ok else "failed", "completes", legs_r.ok, legs_r.reason)
    allr = orbit_s + legs_s + [climb] + orbit_r + [legs_r]
    mx = max(r.max_n for r in allr)
    add("load_factor", round(mx, 2), [None, round(veh.n_ult, 2)], mx <= veh.n_ult, f"peak load in the flown segments vs spar limit {veh.n_ult:.1f} g")
    st = max(r.stalled_s for r in allr)
    add("no_stall", round(st, 2), [None, 0.5], st <= 0.5, "seconds past the stall angle while flying")
    out["ok"] = all(c["pass"] for c in chk)
    # ---- metrics: from the deterministic (scored) segments only
    good_orbits = [r for r in orbit_s if r.ok]
    good_legs = [r for r in legs_s if r.ok]
    m = {}
    if good_orbits:
        p = float(np.mean([r.mean_power for r in good_orbits]))
        m["orbit_power_w"] = p
        m["endurance_min"] = out["usable_wh"] / p * 60.0
    if good_legs:
        whkm = float(np.mean([r.wh_per_km for r in good_legs]))
        m["wh_per_km"] = whkm
        m["range_km"] = out["usable_wh"] / whkm
        m["km_per_wh"] = 1.0 / whkm
    m["vmax_kmh"] = sw["v_max"] * 3.6
    m["climb_ms"] = climb.climb_ms
    m["payload_fraction"] = payload_fraction
    out["metrics"] = m
    key = {"endurance": "endurance_min", "range": "range_km", "speed": "vmax_kmh", "payload": "payload_fraction", "efficiency": "km_per_wh"}[preset]
    out["score"] = float(m.get(key, 0.0)) if out["ok"] else 0.0
    out["score_key"] = key
    return out
