"""2D airfoil properties for NACA 4-digit sections (empirical, low Reynolds number).

All angles in radians unless a name says ``_deg``. Validity: Re 1e5 - 1e6, t/c 0.06 - 0.21,
camber 0 - 6 %. Outside that range the fits are clamped, never extrapolated wildly.

Assumptions (documented, not tuned to any one design):
  * Lift slope  a0 = 2*pi*(1 + 0.77 t/c) * 0.90  per rad. The (1+0.77 t/c) term is the classic
    thickness correction (Abbott & von Doenhoff); the 0.90 factor accounts for viscous
    (boundary-layer decambering) losses at Re ~ 1e5-1e6.
  * Zero-lift angle from thin-airfoil theory, integrated numerically over the NACA 4-digit
    mean line: alpha_L0 = -(1/pi) * int_0^pi dz/dx (cos(theta) - 1) dtheta.
  * CLmax_2D = [1.30 * g(t) + 6 * m] * (Re / 1e6)^0.10, with g(t) = 1 - 8 (t - 0.13)^2
    (peak CLmax near 13 % thickness, falling for thin sections that stall at the leading edge and
    thick ones that stall from the trailing edge). Gives NACA 0012 ~1.15 and NACA 4411 ~1.36 at
    Re 3e5, consistent with low-Re wind-tunnel data (Selig/UIUC, Abbott & von Doenhoff scaled).
  * cd_min = 2 * Cf * (1 + 2 t/c + 60 (t/c)^4), Cf a 40 % laminar / 60 % turbulent blend of
    Blasius (1.328/sqrt(Re)) and Prandtl (0.074/Re^0.2) flat-plate values, plus a laminar
    separation-bubble penalty 0.002 * (1e5/Re)^0.5 that matters only at the low end.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

RE_MIN, RE_MAX = 5e4, 3e6


@dataclass
class Airfoil:
    name: str
    camber: float            # max camber / chord (m)
    camber_pos: float        # position of max camber / chord (p)
    thickness: float         # t/c
    source: str = "naca"     # naca | geometry | default


def parse_naca(name: Optional[str], thickness_ratio: float = 0.12, camber: float = 0.0) -> Airfoil:
    """Parse "NACA4411" / "naca 2410" / "4411". Falls back to the given thickness and camber
    (camber position assumed 0.4c) when the name is missing or not a 4-digit NACA code."""
    if name:
        m = re.search(r"(\d{4})(?!\d)", str(name))
        if m:
            d = m.group(1)
            mc, p, t = int(d[0]) / 100.0, int(d[1]) / 10.0, int(d[2:]) / 100.0
            if t > 0:
                if mc > 0 and p == 0:
                    p = 0.4
                return Airfoil(f"NACA{d}", mc, p, t, "naca")
    t = thickness_ratio if (thickness_ratio and math.isfinite(thickness_ratio) and thickness_ratio > 0) else 0.12
    c = camber if (camber is not None and math.isfinite(camber) and camber >= 0) else 0.0
    return Airfoil(name or "generic", float(min(c, 0.09)), 0.4, float(min(max(t, 0.03), 0.3)),
                   "geometry" if name is None else "default")


def lift_slope_2d(af: Airfoil) -> float:
    """Section lift-curve slope, per rad."""
    return 2.0 * math.pi * (1.0 + 0.77 * af.thickness) * 0.90


_THETA = np.linspace(0.0, math.pi, 201)


def zero_lift_angle(af: Airfoil) -> float:
    """Thin-airfoil zero-lift angle (rad, negative for positive camber)."""
    m, p = af.camber, af.camber_pos
    if m <= 0 or p <= 0 or p >= 1:
        return 0.0
    x = 0.5 * (1.0 - np.cos(_THETA))
    dzdx = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return float(-np.trapezoid(dzdx * (np.cos(_THETA) - 1.0), _THETA) / math.pi)


def _clamp_re(re_: float) -> float:
    if not (re_ and math.isfinite(re_)):
        return 3e5
    return min(max(re_, RE_MIN), RE_MAX)


def cl_max_2d(af: Airfoil, re_: float) -> float:
    """Section CLmax at Reynolds number re_ (empirical fit, see module docstring)."""
    t = min(max(af.thickness, 0.04), 0.25)
    g = 1.0 - 8.0 * (t - 0.13) ** 2
    base = 1.30 * g + 6.0 * min(af.camber, 0.07)
    return float(min(max(base * (_clamp_re(re_) / 1e6) ** 0.10, 0.5), 1.9))


def skin_friction_2d(re_: float) -> float:
    """Flat-plate Cf, 40 % laminar / 60 % turbulent blend (per side)."""
    r = _clamp_re(re_)
    return 0.4 * 1.328 / math.sqrt(r) + 0.6 * 0.074 / r**0.2


def cd_min(af: Airfoil, re_: float) -> float:
    """Minimum section profile drag."""
    t = af.thickness
    ff = 1.0 + 2.0 * t + 60.0 * t**4
    bubble = 0.002 * math.sqrt(1e5 / _clamp_re(re_))
    return float(2.0 * skin_friction_2d(re_) * ff + bubble)
