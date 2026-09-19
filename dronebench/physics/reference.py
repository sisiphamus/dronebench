"""Hand-built PhysicsInput for the Titan Falcon V2 (calibration reference, no CAD needed).

Published: span 2093 mm, wing area 45.14 dm^2, NACA 4411 root -> 3411 tip, twin 2816 motors,
~3.5 kg AUW, ~1.6 Wh/km at 45-65 km/h. Planform assumption: constant 0.26 m chord to
y = 0.602 m, then straight taper to 0.051 m at the tip (gives S = 0.4514 m^2).
Tail, fuselage, nacelle and servo numbers are estimates from photos (documented inline).
"""
from __future__ import annotations

import math

import numpy as np

from ..model import (AeroGeometry, BatterySpec, ControlSurfaceGeom, Material, MassProps, MotorSpec,
                     PhysicsInput, PowerTrain, PropSpec, ServoSpec, SparSpec, SurfaceGeom)

CF_TUBE = Material("CF tube", 1550.0, "composite", source="assumed")


def _rect_surface(name, span, chord, x_le, airfoil, t, camber=0.0, ar=None, z=0.0) -> SurfaceGeom:
    area = span * chord
    return SurfaceGeom(name=name, span_m=span, area_m2=area, mac_m=chord, root_chord_m=chord, tip_chord_m=chord,
                       aspect_ratio=ar if ar is not None else span**2 / area, taper=1.0, sweep_le_deg=0.0,
                       dihedral_deg=0.0, x_le_root_m=x_le, x_ac_m=x_le + 0.25 * chord, z_m=z,
                       airfoil=airfoil, thickness_ratio=t, camber=camber)


def falcon_wing(x_le_root: float = 0.30) -> SurfaceGeom:
    b2, y_kink, c_r, c_t, sweep = 2.093 / 2, 0.602, 0.26, 0.051, math.radians(3.0)
    y = np.linspace(0.0, b2, 401)
    c = np.where(y <= y_kink, c_r, c_r + (c_t - c_r) * (y - y_kink) / (b2 - y_kink))
    x_le = x_le_root + y * math.tan(sweep)
    S = 2 * np.trapezoid(c, y)
    mac = 2 / S * np.trapezoid(c**2, y)
    x_ac = 2 / S * np.trapezoid(c * (x_le + 0.25 * c), y)
    stations = [(float(yy), float(cc), float(xx), 0.0) for yy, cc, xx in zip(y[::40], c[::40], x_le[::40])]
    return SurfaceGeom(name="wing", span_m=2 * b2, area_m2=float(S), mac_m=float(mac), root_chord_m=c_r,
                       tip_chord_m=c_t, aspect_ratio=(2 * b2) ** 2 / float(S), taper=c_t / c_r, sweep_le_deg=3.0,
                       dihedral_deg=0.5, x_le_root_m=x_le_root, x_ac_m=float(x_ac), z_m=0.0, airfoil="NACA4411",
                       thickness_ratio=0.11, camber=0.04, stations=stations)


def falcon_v2_input(mass_kg: float = 3.5, cg_frac_mac: float = 0.28, cruise_kmh: float | None = 55.0) -> PhysicsInput:
    wing = falcon_wing()
    # htail NACA 2410, 0.55 m x 0.136 m = 0.075 m^2, ac 0.75 m aft of wing ac
    ht_c, ht_b = 0.136, 0.55
    htail = _rect_surface("htail", ht_b, ht_c, wing.x_ac_m + 0.75 - 0.25 * ht_c, "NACA2410", 0.10, 0.02)
    # vtail 0.25 m high x 0.16 m = 0.04 m^2 (symmetric), ac at the same station
    vt_c, vt_h = 0.16, 0.25
    vtail = _rect_surface("vtail", vt_h, vt_c, wing.x_ac_m + 0.75 - 0.25 * vt_c, "NACA0010", 0.10, ar=vt_h**2 / (vt_h * vt_c))
    geom = AeroGeometry(
        wing=wing, htail=htail, vtail=vtail, fuselage_length_m=1.2, fuselage_max_diam_m=0.14,
        # two motor nacelles ~0.20 m x 0.05 m: 0.8 pi d L each
        wetted_area_m2={"nacelle": 2 * 0.8 * math.pi * 0.05 * 0.20},
        control_surfaces=[
            ControlSurfaceGeom("aileron_L", "aileron", 0.45 * 0.055, 0.055, 0.45, "wing"),
            ControlSurfaceGeom("aileron_R", "aileron", 0.45 * 0.055, 0.055, 0.45, "wing"),
            ControlSurfaceGeom("flap_L", "flap", 0.40 * 0.065, 0.065, 0.40, "wing"),
            ControlSurfaceGeom("flap_R", "flap", 0.40 * 0.065, 0.065, 0.40, "wing"),
            ControlSurfaceGeom("elevator", "elevator", 0.55 * 0.045, 0.045, 0.55, "htail"),
            ControlSurfaceGeom("rudder", "rudder", 0.22 * 0.06, 0.06, 0.22, "vtail"),
        ])
    x_le_mac = wing.x_ac_m - 0.25 * wing.mac_m
    mass = MassProps(total_kg=mass_kg, cg_m=np.array([x_le_mac + cg_frac_mac * wing.mac_m, 0.0, 0.0]))
    trains = [PowerTrain(MotorSpec(f"motor_{s}", kv=620, rm_ohm=0.071, i0_a=0.8, imax_a=35, mass_g=128),
                         PropSpec(f"prop_{s}", 11, 8, 2), esc_imax_a=40) for s in ("L", "R")]
    # Molicel P42A 6S2P: 4.2 Ah / 45 A per cell -> 8.4 Ah pack, ~10.7 C continuous
    battery = BatterySpec("battery", cells_s=6, capacity_mah=8400, c_rating=10.7, chemistry="liion",
                          ir_mohm_per_cell=15.0, mass_g=840, cells_p=2)
    spars = [SparSpec("spar_main", 10, 8, 2.0, CF_TUBE, "main_spar"),
             SparSpec("spar_rear", 8, 6, 1.6, CF_TUBE, "rear_spar")]
    servos = [ServoSpec(f"servo_{c.id}", 3.5, c.id) for c in geom.control_surfaces]
    return PhysicsInput(geom=geom, mass=mass, trains=trains, battery=battery, spars=spars, servos=servos,
                        payload_kg=0.0, cruise_ms=cruise_kmh / 3.6 if cruise_kmh else None)
