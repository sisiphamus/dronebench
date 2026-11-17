"""Cascaded autopilot whose gains are computed from the design's own aerodynamic derivatives.

Every design gets a competent, comparable controller (the same method every time), so a design that
cannot be flown by it is a design problem, not a tuning problem. Structure (Beard & McLain, "Small
Unmanned Aircraft"):

    course -> bank angle (roll hold, PD on aileron)        orbit / straight-line path following
    altitude -> pitch angle (pitch hold, PD on elevator)
    airspeed -> throttle (PI on top of the trimmed throttle)
    sideslip and yaw rate -> rudder (turn coordination + yaw damper)

Gains come from the linearised short-period (pitch) and roll dynamics: for a loop x'' = a1 x' + a2 x + b u,
PD gains put it at a chosen natural frequency and damping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import flight as Fl


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Command:
    course: float = 0.0          # rad, ground course
    alt: float = 100.0           # m
    airspeed: float = 15.0       # m/s
    phi_ff: float = 0.0          # feed-forward bank for a steady orbit


class Autopilot:
    def __init__(self, veh: Fl.Vehicle, V: float, alt: float = 100.0, phi_max_deg: float = 30.0):
        self.v = veh
        self.V = V
        tr = Fl.trim(veh, V, alt=alt)
        self.tr = tr
        self.phi_max = math.radians(phi_max_deg)
        rho = Fl._rho(alt)
        q = 0.5 * rho * V * V
        t = veh.tab
        S, b, c = t.s_ref, t.b_ref, t.c_ref
        I = veh.I
        d = t.d
        self.Lp = q * S * b * b / (2 * V * I[0, 0]) * d["Clp"]
        self.Lda = q * S * b / I[0, 0] * d["Clda"]
        self.Mq = q * S * c * c / (2 * V * I[1, 1]) * d["Cmq"]
        self.Ma = q * S * c / I[1, 1] * d["Cma"]
        self.Mde = q * S * c / I[1, 1] * d["Cmde"]
        self.ok_controls = abs(self.Lda) > 1e-3 and abs(self.Mde) > 1e-3
        # roll loop: bandwidth limited by the servo (tau 0.04 s)
        wn, z = 5.0, 0.9
        self.kp_phi = wn * wn / max(abs(self.Lda), 1e-3)
        self.kd_phi = max((2 * z * wn + self.Lp) / max(abs(self.Lda), 1e-3), 0.0)
        # pitch loop (attitude): theta'' = Mq theta' + Ma theta + Mde de
        wn_t, z_t = max(4.5, 1.3 * math.sqrt(max(-self.Ma, 0.0))), 0.8
        self.kp_th = (wn_t ** 2 + self.Ma) / self.Mde if self.Mde != 0 else 0.0
        self.kd_th = (2 * z_t * wn_t + self.Mq) / self.Mde if self.Mde != 0 else 0.0
        self.kp_th = float(np.clip(self.kp_th, -3.0, 3.0)); self.kd_th = float(np.clip(self.kd_th, -2.0, 2.0))
        # outer loops
        self.k_chi = 1.4            # bank per rad of course error
        self.k_h = 0.045            # pitch per metre of altitude error (rad/m)
        self.ki_h = 0.003
        self.k_V = 0.10             # throttle per m/s
        self.ki_V = 0.03
        self.int_h = 0.0
        self.int_V = 0.0
        self.theta_trim = tr["alpha"]
        self.de_trim = tr["de"]
        self.thr_trim = tr["throttle"]
        self.r_filt = 0.0

    def update(self, o: dict, cmd: Command, dt: float) -> Fl.Controls:
        phi, theta = o["phi"], o["theta"]
        # --- lateral
        phi_c = float(np.clip(cmd.phi_ff + self.k_chi * wrap(cmd.course - o["course"]), -self.phi_max, self.phi_max))
        da = self.kp_phi * (phi_c - phi) - self.kd_phi * o["p"]
        # coordinated turn: rudder drives sideslip to zero, plus yaw damping on the washed-out rate
        r_turn = Fl.G * math.tan(phi) / max(o["airspeed"], 5.0)
        dr = 0.9 * o["beta"] - 0.35 * (o["r"] - r_turn)
        # --- longitudinal
        eh = cmd.alt - o["alt"]
        self.int_h = float(np.clip(self.int_h + eh * dt, -80.0, 80.0))
        th_c = self.theta_trim + float(np.clip(self.k_h * eh + self.ki_h * self.int_h, -0.30, 0.30))
        de = self.de_trim + self.kp_th * (th_c - theta) - self.kd_th * o["q"]
        eV = cmd.airspeed - o["airspeed"]
        self.int_V = float(np.clip(self.int_V + eV * dt, -10.0, 10.0))
        thr = self.thr_trim + self.k_V * eV + self.ki_V * self.int_V + 0.6 * math.sin(max(th_c - self.theta_trim, 0.0))
        return Fl.Controls(da=da, de=de, dr=dr, thr=float(np.clip(thr, 0.0, 1.0)))


def _update_climb(self, o: dict, cmd: Command, dt: float) -> Fl.Controls:
    """Full throttle; pitch holds the commanded airspeed (this is how a real aircraft is flown at best climb)."""
    phi, theta = o["phi"], o["theta"]
    phi_c = float(np.clip(self.k_chi * wrap(cmd.course - o["course"]), -self.phi_max, self.phi_max))
    da = self.kp_phi * (phi_c - phi) - self.kd_phi * o["p"]
    r_turn = Fl.G * math.tan(phi) / max(o["airspeed"], 5.0)
    dr = 0.9 * o["beta"] - 0.35 * (o["r"] - r_turn)
    eV = o["airspeed"] - cmd.airspeed
    self.int_V = float(np.clip(self.int_V + eV * dt, -10.0, 10.0))
    th_c = self.theta_trim + float(np.clip(0.06 * eV + 0.01 * self.int_V, -0.05, 0.45))
    de = self.de_trim + self.kp_th * (th_c - theta) - self.kd_th * o["q"]
    return Fl.Controls(da=da, de=de, dr=dr, thr=1.0)


Autopilot.update_climb = _update_climb


# ---------------------------------------------------------------- guidance
def orbit_course(o: dict, center, radius: float, direction: int = 1, k: float = 4.0):
    """Course command and bank feed-forward to fly a circle (direction +1 = clockwise seen from above)."""
    pn, pe = o["pos"][0] - center[0], o["pos"][1] - center[1]
    d = math.hypot(pn, pe)
    ang = math.atan2(pe, pn)
    chi = ang + direction * (math.pi / 2 + math.atan(k * (d - radius) / radius))
    V = max(o["gs"], 5.0)
    phi_ff = direction * math.atan(V * V / (Fl.G * radius)) * (1.0 if d < 2 * radius else 0.0)
    return wrap(chi), phi_ff


def line_course(o: dict, p0, p1, k: float = 0.02, chi_inf: float = math.radians(60)):
    """Course command to follow the line p0 -> p1; also returns cross-track error and along-track fraction."""
    q = np.array([p1[0] - p0[0], p1[1] - p0[1]])
    L = float(np.linalg.norm(q)) or 1.0
    q = q / L
    chi_q = math.atan2(q[1], q[0])
    rel = np.array([o["pos"][0] - p0[0], o["pos"][1] - p0[1]])
    along = float(rel @ q)
    e_y = float(-math.sin(chi_q) * rel[0] + math.cos(chi_q) * rel[1])
    chi = chi_q - chi_inf * 2 / math.pi * math.atan(k * e_y)
    return wrap(chi), e_y, along / L
