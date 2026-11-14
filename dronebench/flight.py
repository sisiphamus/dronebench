"""Six-degree-of-freedom flight simulation of a DroneBench design.

    bits = pipeline.core(design, preset)              # mass, geometry, powertrain from the design
    veh  = Vehicle.from_bits(bits, design)            # + aero table (AeroSandbox), inertia, tail incidence
    sim  = Sim(veh, Wind(mean=(3, 0, 0), turb=1.0, seed=1))
    sim.reset(alt=100, airspeed=15)
    obs  = sim.step(Controls(da, de, dr, thr), dt)    # surfaces in rad, throttle 0..1

Equations: rigid body, body axes x forward / y right / z down, NED navigation frame, unit-quaternion attitude,
RK4 (compiled with numba, `_dyn.py`). Forces: gravity, aerodynamics (`aero.AeroTable`), propeller thrust
(motor + prop + battery with voltage sag and state of charge). Wind: steady vector + Dryden turbulence.
Servos have first-order lag, a deflection limit and a rate limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import _dyn as K
from . import aero as A
from .physics import propulsion as P

G = 9.80665
RHO0 = 1.225


def _rho(alt_m: float) -> float:
    return RHO0 * math.exp(-alt_m / 9300.0)


# ---------------------------------------------------------------- quaternions
def quat_to_dcm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])   # body -> NED


def quat_from_euler(phi, theta, psi):
    cp, sp = math.cos(phi / 2), math.sin(phi / 2)
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    return np.array([cp * ct * cy + sp * st * sy, sp * ct * cy - cp * st * sy,
                     cp * st * cy + sp * ct * sy, cp * ct * sy - sp * st * cy])


def euler_from_quat(q):
    w, x, y, z = q
    phi = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    theta = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    psi = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return phi, theta, psi


# ---------------------------------------------------------------- wind
@dataclass
class Wind:
    mean: tuple = (0.0, 0.0, 0.0)       # NED m/s, direction the air moves toward
    turb: float = 0.0                    # 0 none, 1 light (sigma_u,v 1.06 and sigma_w 0.7 m/s), 2 moderate, ...
    seed: int = 0


GUST_L = np.array([200.0, 200.0, 50.0])         # Dryden length scales, low altitude (Beard & McLain)
GUST_S = np.array([1.06, 1.06, 0.7])


# ---------------------------------------------------------------- battery and powertrain
_LIPO = np.array([[0.0, 3.00], [0.05, 3.35], [0.2, 3.62], [0.5, 3.78], [0.8, 3.98], [1.0, 4.15]])
_LIION = np.array([[0.0, 2.90], [0.05, 3.20], [0.2, 3.45], [0.5, 3.62], [0.8, 3.85], [1.0, 4.10]])


class Powertrain:
    """Motor + prop + ESC + battery. Trains are treated as identical and share the pack equally.

    The thrust and current of a train depend only on motor voltage and airspeed, so they are tabulated once
    (vectorised bisection on rpm) and interpolated inside the compiled loop."""

    def __init__(self, trains, battery, usable: float = 0.85, esc_eff: float = 0.95):
        self.trains = trains
        self.batt = battery
        self.n_trains = max(len(trains), 1)
        self.pc = [P.prop_coeffs(t.prop.diam_in, t.prop.pitch_in, t.prop.blades) for t in trains]
        self.cap_wh = P.battery_energy_wh(battery) if battery else 0.0
        self.usable = usable
        self.esc_eff = esc_eff
        chem = ((battery.chemistry if battery else "") or "").lower().replace("-", "")
        self.curve = _LIION if chem == "liion" else _LIPO
        self.r = P.battery_resistance(battery) if battery else 0.0
        self.cells = max(int(battery.cells_s), 1) if battery else 1
        self._tabs = None

    def tables(self):
        if self._tabs is None:
            if not self.trains:
                z = np.zeros((2, 2))
                self._tabs = (np.array([0.0, 1.0]), np.array([0.0, 1.0]), z, z)
            else:
                tr, pc = self.trains[0], self.pc[0]
                vmax = self.cells * self.curve[-1, 1] * 1.02
                vms = np.linspace(0.0, vmax, 41)
                vas = np.concatenate([np.linspace(0.0, 12.0, 7), np.linspace(14.0, 50.0, 19)])
                VM, VA = np.meshgrid(vms, vas, indexing="ij")
                lo, hi = np.zeros_like(VM), np.full_like(VM, 400.0)
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    _, v = P.motor_state(tr, pc, mid, VA, RHO0)
                    over = v > VM
                    hi = np.where(over, mid, hi)
                    lo = np.where(over, lo, mid)
                n = 0.5 * (lo + hi)
                i, _ = P.motor_state(tr, pc, n, VA, RHO0)
                T = np.asarray(P.prop_thrust(pc, n, VA, RHO0))
                self._tabs = (vms, vas, np.maximum(T, 0.0), np.maximum(i, 0.0) * (n > 1e-3))
        return self._tabs

    def thrust_at(self, vm: float, V: float, rho: float = RHO0) -> float:
        vms, vas, T, _ = self.tables()
        return self.n_trains * float(K._bilin(T, vms, vas, vm, V)) * rho / RHO0


# ---------------------------------------------------------------- vehicle
@dataclass
class Vehicle:
    mass: float
    cg_d: np.ndarray                      # design frame (x aft, y right, z up), m
    I: np.ndarray                         # inertia about CG, body axes, kg m^2
    tab: A.AeroTable
    pt: Powertrain
    prop_pos_b: np.ndarray                # thrust application point relative to CG, body axes
    n_ult: float = 8.0                    # structural limit load factor (ultimate)
    cl_max: float = 1.4
    spar_sf_35g: float = 2.0
    cruise_ms: float = 15.0
    notes: list = field(default_factory=list)
    ref_b: np.ndarray = field(default_factory=lambda: np.zeros(3))   # aero reference point relative to CG, body axes
    inc_off: float = 0.0                  # tail incidence, as an equivalent elevator bias (rad)
    v_design: float = 15.0
    cda_extra: float = 0.0                # parasite drag area of external bodies, m^2
    cd0_factor: float = 1.0               # calibration factor on drag (gaps, seams, roughness); 1.0 = AeroSandbox as is
    _packed: tuple = None

    @staticmethod
    def from_bits(bits, design, cd0_factor: float = 1.0, v_ref: float = 15.0) -> "Vehicle":
        """From pipeline.core(...) output (`bits`) plus the Design. Nothing here is hard-coded per aircraft."""
        mass, geom, pin, pres = bits["mass"], bits["geom"], bits["pin"], bits["pres"]
        cg = np.asarray(mass.cg_m, float)
        tab = A.table_for(geom, cg, v_ref=v_ref)
        I = inertia_body(design, mass, cg)
        ref_b = np.array([-(tab.ref[0] - cg[0]), tab.ref[1] - cg[1], -(tab.ref[2] - cg[2])])
        motors = [m for m in design.of_type("motor") if m.cg_m is not None]
        if motors:
            pos = np.mean([np.asarray(m.cg_m, float) for m in motors], axis=0)
            prop_b = np.array([-(pos[0] - cg[0]), pos[1] - cg[1] if len(motors) == 1 else 0.0, -(pos[2] - cg[2])])
        else:
            prop_b = np.zeros(3)
        pt = Powertrain(pin.trains, pin.battery, usable=pin.usable_battery_fraction)
        sf = float(pres.spar_sf) if pres is not None and np.isfinite(pres.spar_sf) else 2.0
        v_design = float(pin.cruise_ms) if getattr(pin, "cruise_ms", None) else 15.0
        veh = Vehicle(mass=float(mass.total_kg), cg_d=cg, I=I, tab=tab, pt=pt, prop_pos_b=prop_b,
                      n_ult=max(3.5 * sf, 1.0), cl_max=tab.cl_max, spar_sf_35g=sf, cruise_ms=v_design,
                      ref_b=ref_b, v_design=v_design, cda_extra=A.extra_drag_cda(geom), cd0_factor=cd0_factor)
        # Tail incidence: a builder sets it so the elevator sits at neutral in design cruise. A CG that moves
        # afterwards (or any other edit) shows up as elevator trim drag, as it does on a real aircraft.
        tr = trim(veh, v_design)
        veh.inc_off = tr["de"]
        veh._packed = None
        return veh

    # ---- numba parameter pack
    def pack(self):
        if self._packed is None:
            t = self.tab
            Pv = np.zeros(K.NP)
            Pv[K.P_MASS] = self.mass; Pv[K.P_S] = t.s_ref; Pv[K.P_B] = t.b_ref; Pv[K.P_C] = t.c_ref
            Pv[K.P_REF:K.P_REF + 3] = self.ref_b
            Pv[K.P_PROP:K.P_PROP + 3] = self.prop_pos_b
            Pv[K.P_I:K.P_I + 9] = self.I.reshape(-1)
            Pv[K.P_IINV:K.P_IINV + 9] = np.linalg.inv(self.I).reshape(-1)
            Pv[K.P_ASTALL] = t.alpha_stall; Pv[K.P_CD0F] = self.cd0_factor; Pv[K.P_CDAX] = self.cda_extra
            Pv[K.P_INC] = self.inc_off
            pt = self.pt
            Pv[K.P_NTR] = pt.n_trains; Pv[K.P_ESC] = pt.esc_eff; Pv[K.P_CAP] = pt.cap_wh; Pv[K.P_RBAT] = pt.r
            Pv[K.P_CELLS] = pt.cells; Pv[K.P_USABLE] = pt.usable
            Pv[K.P_DEFL] = DEFL_MAX; Pv[K.P_RATE] = SERVO_RATE; Pv[K.P_TAU] = SERVO_TAU; Pv[K.P_TAUP] = 0.08
            d = t.d
            Dv = np.array([d["CLq"], d["CLde"], d["CYb"], d["CYp"], d["CYr"], d["CYdr"], d["Clb"], d["Clp"], d["Clr"],
                           d["Clda"], d["Cldr"], d["Cmq"], d["Cmde"], d["Cnb"], d["Cnp"], d["Cnr"], d["Cnda"], d["Cndr"],
                           d.get("CDde", 0.0), d.get("CDda", 0.0)])
            vms, vas, T, I = pt.tables()
            c = np.ascontiguousarray
            self._packed = (Pv, Dv, c(t.alphas), np.log(c(t.speeds)), c(t.CL), c(t.CD), c(t.Cm), c(T), c(vms), c(vas),
                            c(T), c(I), c(pt.curve[:, 0]), c(pt.curve[:, 1]))
        return self._packed


def inertia_body(design, mass, cg) -> np.ndarray:
    """Inertia tensor about the CG in body axes from the part masses (point mass + box about its own CG)."""
    I = np.zeros((3, 3))
    for pid, g in mass.by_part_g.items():
        p = design.parts.get(pid)
        if p is None or p.cg_m is None or g <= 0:
            continue
        m = g / 1000.0
        r = np.asarray(p.cg_m, float) - cg
        rb = np.array([-r[0], r[1], -r[2]])          # design frame -> body frame (x, z flip)
        I += m * ((rb @ rb) * np.eye(3) - np.outer(rb, rb))
        bb = getattr(p, "bbox_m", None)
        if bb is not None:
            d = np.abs(np.asarray(bb[1], float) - np.asarray(bb[0], float))
            I += np.diag([m / 12 * (d[1] ** 2 + d[2] ** 2), m / 12 * (d[0] ** 2 + d[2] ** 2), m / 12 * (d[0] ** 2 + d[1] ** 2)])
    w = np.linalg.eigvalsh(I)
    if w.min() <= 1e-6:
        I += np.eye(3) * (1e-4 - min(w.min(), 0))
    return I


# ---------------------------------------------------------------- simulator
@dataclass
class Controls:
    da: float = 0.0          # roll right, rad
    de: float = 0.0          # pitch up, rad
    dr: float = 0.0          # yaw right, rad
    thr: float = 0.0         # 0..1


DEFL_MAX = math.radians(25.0)
SERVO_RATE = math.radians(240.0)     # rad/s
SERVO_TAU = 0.04


class Sim:
    """Aircraft in the air. State: pos NED, body velocity (ground-relative), quaternion, body rates."""

    def __init__(self, veh: Vehicle, wind: Optional[Wind] = None, dt: float = 0.01):
        self.v = veh
        self.dt = dt
        self.wind = wind or Wind()
        self.reset()

    def reset(self, alt: float = 100.0, airspeed: float = 15.0, heading: float = 0.0, pos=(0.0, 0.0), pitch=None, bank=0.0,
              soc: float = 1.0):
        v = self.v
        self.rng = np.random.default_rng(self.wind.seed)
        self.gust_s = GUST_S * self.wind.turb
        self.wind_mean = np.array(self.wind.mean, float)
        theta = 0.04 if pitch is None else pitch
        q = quat_from_euler(bank, theta, heading)
        R = quat_to_dcm(q)
        vb = np.array([airspeed * math.cos(0.05), 0.0, airspeed * math.sin(0.05)]) + R.T @ self.wind_mean
        self.x = np.concatenate([[pos[0], pos[1], -alt], vb, q, np.zeros(3)])
        self.surf = np.zeros(3)
        cap = v.pt.cap_wh
        self.pt_state = np.array([0.0, 0.0, (1.0 - soc) * cap, 0.0])      # thrust, current, used Wh, battery A
        self.gust = np.zeros(3)
        self.t = 0.0
        self.n_load = 1.0
        self.max_n, self.min_n = 1.0, 1.0
        self.stalled_time = 0.0
        self.crashed = False
        self.structural_fail = False
        self.thrust = 0.0
        self.power = 0.0
        self.alpha = 0.0
        self._soc = soc
        self._prime(airspeed)       # start the powertrain at trim so the first seconds are not a spin-up transient
        return self.obs()

    def _prime(self, V):
        Pv, Dv, al, ls, CL, CD, Cm, _, vms, vas, T, I, bs, bv = self.v.pack()
        vterm = self.v.pt.cells * float(np.interp(self._soc, bs, bv))
        tr = trim(self.v, V, alt=-self.x[2])
        vm = tr["throttle"] * 0.98 * vterm
        self.pt_state[0] = float(K._bilin(T, vms, vas, vm, V))
        self.pt_state[1] = float(K._bilin(I, vms, vas, vm, V))

    # ---- derived
    def euler(self):
        return euler_from_quat(self.x[6:10])

    def air_velocity_body(self):
        R = quat_to_dcm(self.x[6:10])
        return self.x[3:6] - R.T @ (self.wind_mean + self.gust)

    def airspeed(self):
        return float(np.linalg.norm(self.air_velocity_body()))

    def alt(self):
        return -float(self.x[2])

    def used_wh(self):
        return float(self.pt_state[2])

    @property
    def soc(self):
        return max(0.0, 1.0 - self.pt_state[2] / max(self.v.pt.cap_wh, 1e-9))

    def depleted(self):
        pt = self.v.pt
        voc_cell = float(np.interp(self.soc, pt.curve[:, 0], pt.curve[:, 1]))
        return self.pt_state[2] >= pt.usable * pt.cap_wh or voc_cell <= 3.15

    def obs(self):
        q = self.x[6:10]
        phi, theta, psi = euler_from_quat(q)
        va = self.air_velocity_body()
        V = float(np.linalg.norm(va))
        alpha = math.atan2(va[2], max(va[0], 1e-3))
        beta = math.asin(max(-1.0, min(1.0, va[1] / max(V, 1e-3))))
        gv = quat_to_dcm(q) @ self.x[3:6]
        return {"pos": self.x[0:3].copy(), "alt": self.alt(), "airspeed": V, "alpha": alpha, "beta": beta, "phi": phi,
                "theta": theta, "psi": psi, "p": float(self.x[10]), "q": float(self.x[11]), "r": float(self.x[12]), "gv": gv,
                "course": math.atan2(gv[1], gv[0]), "gs": math.hypot(gv[0], gv[1]), "soc": self.soc, "n": self.n_load,
                "t": self.t, "power": self.power, "thrust": self.thrust, "surf": self.surf.copy()}

    def _forces(self, vb, w, q, surf, thrust, gust):
        """(F, M, alpha, V) in body axes about the CG; python wrapper over the compiled function."""
        Pv, Dv, al, ls, CL, CD, Cm = self.v.pack()[:7]
        wind = self.wind_mean + gust
        return K._forces(np.asarray(vb, float), np.asarray(w, float), np.asarray(q, float), np.asarray(surf, float), float(thrust),
                         wind, Pv, Dv, al, ls, CL, CD, Cm, -float(self.x[2]))

    def step(self, ctl: Controls, dt: Optional[float] = None):
        dt = dt or self.dt
        v = self.v
        Pv, Dv, al, ls, CL, CD, Cm, _, vms, vas, T, I, bs, bv = v.pack()
        cmd = np.array([ctl.da, ctl.de, ctl.dr, ctl.thr], float)
        noise = self.rng.standard_normal(3) if self.wind.turb > 0 else np.zeros(3)
        n, V, alpha, thrust, p_batt, soc = K.step(self.x, self.surf, self.pt_state, self.gust, cmd, noise, dt, Pv, Dv, al, ls,
                                                  CL, CD, Cm, self.wind_mean, T, vms, vas, T, I, bs, bv, self.gust_s, GUST_L)
        self.t += dt
        self.n_load, self.thrust, self.power, self.alpha = float(n), float(thrust), float(p_batt), float(alpha)
        self.max_n = max(self.max_n, n); self.min_n = min(self.min_n, n)
        if n > v.n_ult or n < -0.4 * v.n_ult:
            self.structural_fail = True
        if abs(alpha) > v.tab.alpha_stall:
            self.stalled_time += dt
        if self.x[2] > 0.0 or not np.all(np.isfinite(self.x)):
            self.crashed = True
        return self.obs()


# ---------------------------------------------------------------- trim
def throttle_for_thrust(veh: Vehicle, Tt: float, V: float, rho: float) -> float:
    pt = veh.pt
    if not pt.trains:
        return 0.0
    vterm = pt.cells * float(np.interp(0.8, pt.curve[:, 0], pt.curve[:, 1]))     # near-full pack
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if pt.thrust_at(mid * 0.98 * vterm, V, rho) > Tt:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def trim(veh: Vehicle, V: float, gamma: float = 0.0, alt: float = 100.0) -> dict:
    """Steady straight flight at airspeed V and flight-path angle gamma: solves alpha, elevator and thrust."""
    from scipy.optimize import fsolve
    Pv, Dv, al, ls, CL, CD, Cm = veh.pack()[:7]
    z3 = np.zeros(3)

    def residual(u):
        alpha, de, T = u
        theta = gamma + alpha
        vb = np.array([V * math.cos(alpha), 0.0, V * math.sin(alpha)])
        q = quat_from_euler(0.0, theta, 0.0)
        F, M, _, _ = K._forces(vb, z3, q, np.array([0.0, de, 0.0]), float(T), z3, Pv, Dv, al, ls, CL, CD, Cm, alt)
        return [F[0], F[2], M[1]]

    sol, info, ier, msg = fsolve(residual, [0.05, 0.0, 5.0], full_output=True)
    alpha, de, T = (float(x) for x in sol)
    thr = throttle_for_thrust(veh, T, V, _rho(alt))
    return {"alpha": alpha, "de": de, "thrust": T, "throttle": thr, "ok": bool(ier == 1 and thr < 0.999),
            "residual": float(np.abs(residual([alpha, de, T])).max())}
