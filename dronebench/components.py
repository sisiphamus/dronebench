"""Catalog of common COTS parts (motors, props, servos, ESCs, avionics, batteries).

Values are approximate figures from manufacturer datasheets / product pages; every
entry carries a `source_note`. `lookup(part_number)` is fuzzy: case-insensitive,
punctuation-insensitive, and every number in the catalog name must appear in the
query (so "2816 620KV" never matches the 900KV variant).

Entry shape:
  {"name", "type", "specs": {...}, "mass_g", "material": <materials name>, "sub": {part: material},
   "source_note", "aliases": [...]}
"""
from __future__ import annotations

import copy
import re
from typing import Optional

_DS = "manufacturer datasheet (approx)"
_MOTOR_SUB = {"bell": "Al 6061", "shaft": "Steel", "windings": "Copper", "magnets": "NdFeB", "bearings": "Steel"}
_ELEC_SUB = {"pcb": "FR4", "components": "mixed", "wires": "Copper+Silicone"}


def _motor(name, kv, rm, i0, imax, mass, aliases=(), note=_DS, **extra):
    return {"name": name, "type": "motor", "specs": {"kv": kv, "rm_ohm": rm, "i0_a": i0, "imax_a": imax,
                                                      "mass_g": mass, **extra},
            "mass_g": mass, "material": "Motor assembly", "sub": dict(_MOTOR_SUB), "source_note": note,
            "aliases": list(aliases)}


def _servo(name, torque, mass, metal, speed_s=0.1, aliases=(), dims=(23, 12, 24)):
    return {"name": name, "type": "servo",
            "specs": {"torque_kgcm": torque, "voltage_v": 4.8, "speed_s_60deg": speed_s, "mass_g": mass,
                      "gears": "metal" if metal else "nylon", "dims_mm": list(dims)},
            "mass_g": mass, "material": "Nylon PA66",
            "sub": {"case": "Nylon PA66", "gears": "Brass" if metal else "Nylon PA66", "motor": "Copper+NdFeB",
                    "pcb": "FR4"},
            "source_note": _DS, "aliases": list(aliases)}


def _prop(name, d, p, mass, aliases=()):
    return {"name": name, "type": "prop", "specs": {"diam_in": d, "pitch_in": p, "blades": 2, "mass_g": mass},
            "mass_g": mass, "material": "Nylon glass-filled", "sub": {"blades": "Nylon glass-filled"},
            "source_note": "APC product data (approx); APC E props are glass-filled nylon", "aliases": list(aliases)}


def _esc(name, imax, mass, aliases=()):
    return {"name": name, "type": "esc", "specs": {"imax_a": imax, "burst_a": round(imax * 1.3), "mass_g": mass,
                                                  "cells_max": 6},
            "mass_g": mass, "material": "Electronics", "sub": dict(_ELEC_SUB, heatsink="Al 6061"),
            "source_note": _DS, "aliases": list(aliases)}


def _elec(name, typ, mass, aliases=(), **specs):
    return {"name": name, "type": typ, "specs": {"mass_g": mass, **specs}, "mass_g": mass,
            "material": "Electronics", "sub": dict(_ELEC_SUB), "source_note": _DS, "aliases": list(aliases)}


def _batt(name, s, p, cap, c, ir, mass, chem, aliases=()):
    liion = chem == "liion"
    return {"name": name, "type": "battery",
            "specs": {"cells_s": s, "cells_p": p, "capacity_mah": cap, "c_rating": c, "ir_mohm_per_cell": ir,
                      "mass_g": mass, "chemistry": chem, "nominal_v": round(s * (3.6 if liion else 3.7), 1),
                      "energy_wh": round(s * (3.6 if liion else 3.7) * cap / 1000.0, 1)},
            "mass_g": mass, "material": "Li-ion cell" if liion else "LiPo pack",
            "sub": {"cells": "Li-ion NMC 21700" if liion else "LiPo pouch", "casing": "PVC shrink",
                    "wires": "Copper+Silicone"},
            "source_note": ("Molicel P42A datasheet (4200 mAh, 70 g, 45 A cont. per cell = 10.7C) (approx)"
                            if liion else _DS),
            "aliases": list(aliases)}


CATALOG: list[dict] = [
    # ---- motors ----
    _motor("BrotherHobby Avenger 2816 620KV", 620, 0.061, 0.9, 35, 128, ["Avenger 2816 620KV"]),
    _motor("BrotherHobby Avenger 2816 900KV", 900, 0.042, 1.3, 45, 118, ["Avenger 2816 900KV"]),
    _motor("BrotherHobby Avenger 2816 1050KV", 1050, 0.034, 1.6, 50, 118, ["Avenger 2816 1050KV"]),
    _motor("SunnySky X2216 880KV", 880, 0.105, 0.4, 20, 69, ["X2216 880KV", "X2216-880"]),
    _motor("SunnySky X2216 1100KV", 1100, 0.083, 0.6, 26, 72, ["X2216 1100KV", "X2216-1100"]),
    _motor("SunnySky X2216 1250KV", 1250, 0.070, 0.7, 28, 72, ["X2216 1250KV", "X2216-1250"]),
    _motor("T-Motor AT2312 1150KV", 1150, 0.090, 0.8, 25, 60, ["AT2312 1150KV", "Tmotor AT2312 1150KV"]),
    # ---- servos ----
    _servo("Emax ES08MA II", 2.0, 12, True, 0.12, ["ES08MA", "ES08MAII"]),
    _servo("Emax ES08A II", 1.8, 8.5, False, 0.10, ["ES08A", "ES08AII"]),
    _servo("Emax ES3054", 2.2, 13.4, True, 0.10, []),
    _servo("TowerPro MG90S", 2.2, 13.4, True, 0.10, ["MG90S", "MG90"], dims=(22.8, 12.2, 28.5)),
    _servo("TowerPro SG90", 1.8, 9, False, 0.10, ["SG90", "SG-90", "SG 90"], dims=(22.2, 11.8, 31)),
    _servo("Hitec HS-65MG", 2.2, 11.9, True, 0.14, ["HS65MG", "HS-65 MG"]),
    _servo("Corona DS-929MG", 2.5, 12.5, True, 0.10, ["DS929MG", "DS-929 MG", "Corona 929MG"]),
    # ---- props ----
    _prop("APC 9x6E", 9, 6, 15, ["APC 9x6 E", "9x6E"]),
    _prop("APC 10x7E", 10, 7, 20, ["APC 10x7 E", "10x7E"]),
    _prop("APC 11x8E", 11, 8, 28, ["APC 11x8 E", "11x8E"]),
    _prop("APC 12x8E", 12, 8, 33, ["APC 12x8 E", "12x8E"]),
    _prop("APC 13x8E", 13, 8, 40, ["APC 13x8 E", "13x8E"]),
    # ---- ESCs (generic BLHeli_S / BLHeli_32 fixed-wing ESC with BEC) ----
    _esc("Generic BLHeli 30A ESC", 30, 25, ["30A ESC", "ESC 30A"]),
    _esc("Generic BLHeli 35A ESC", 35, 32, ["35A ESC", "ESC 35A"]),
    _esc("Generic BLHeli 40A ESC", 40, 38, ["40A ESC", "ESC 40A"]),
    _esc("Generic BLHeli 60A ESC", 60, 60, ["60A ESC", "ESC 60A"]),
    # ---- flight controllers ----
    _elec("Matek F405-WTE", "fc", 25, ["F405-WTE", "F405 WTE", "Matek F405 WTE"], has_remote_id=False,
          has_baro=True, has_osd=True, servo_outputs=12, input_v=[6, 30]),
    _elec("Matek H743-WING", "fc", 32, ["H743-WING", "H743 WING", "Matek H743 Wing V3"], has_remote_id=False,
          has_baro=True, has_osd=True, servo_outputs=13, input_v=[6, 36]),
    # ---- receivers ----
    _elec("ExpressLRS EP2 receiver", "rx", 1.0, ["ELRS EP2", "EP2", "ELRS receiver", "ExpressLRS EP2"],
          band="2.4GHz", protocol="ELRS"),
    _elec("FrSky R-XSR", "rx", 1.5, ["RXSR", "R-XSR", "FrSky RXSR"], band="2.4GHz", protocol="ACCST"),
    # ---- GPS, sensors ----
    _elec("Matek M10Q-5883", "gps", 9, ["M10Q-5883", "M10Q 5883", "Matek M10Q", "M10Q"], compass=True),
    _elec("Matek ASPD-4525", "sensor", 6, ["ASPD-4525", "ASPD4525", "pitot 4525", "Matek pitot"],
          kind="airspeed", range_ms=100),
    _elec("Holybro Remote ID", "remote_id", 12, ["Remote ID module", "Holybro RID", "Remote ID"],
          has_remote_id=True, standard="ASTM F3411 broadcast"),
    # ---- cameras ----
    _elec("RunCam Split 4", "camera", 16, ["Split 4", "RunCam Split"], resolution="4K"),
    _elec("Caddx Ratel 2", "camera", 5.9, ["Ratel 2", "Caddx Ratel"], resolution="1200TVL"),
    _elec("RunCam 5 Orange", "camera", 56, ["RunCam 5"], resolution="4K"),
    _elec("Sony RX0 II", "camera", 132, ["RX0 II", "RX0II", "RX0"], resolution="15MP", has_lens_glass=True),
    # ---- batteries ----
    _batt("6S2P Molicel P42A Li-ion", 6, 2, 8400, 10, 15, 900, "liion", ["6S2P P42A", "P42A 6S2P"]),
    _batt("6S1P Molicel P42A Li-ion", 6, 1, 4200, 10, 15, 450, "liion", ["6S1P P42A", "P42A 6S1P"]),
    _batt("4S2P Molicel P42A Li-ion", 4, 2, 8400, 10, 15, 600, "liion", ["4S2P P42A", "P42A 4S2P"]),
    _batt("4S1P Molicel P42A Li-ion", 4, 1, 4200, 10, 15, 300, "liion", ["4S1P P42A", "P42A 4S1P"]),
    _batt("3S 2200mAh 30C LiPo", 3, 1, 2200, 30, 8, 180, "lipo", ["3S 2200 LiPo", "3S2200", "3S 2200mAh"]),
    _batt("3S 1300mAh 45C LiPo", 3, 1, 1300, 45, 10, 110, "lipo", ["3S 1300 LiPo", "3S1300", "3S 1300mAh"]),
    _batt("4S 3300mAh 30C LiPo", 4, 1, 3300, 30, 6, 360, "lipo", ["4S 3300 LiPo", "4S3300", "4S 3300mAh"]),
    _batt("6S 5000mAh 30C LiPo", 6, 1, 5000, 30, 4, 750, "lipo", ["6S 5000 LiPo", "6S5000", "6S 5000mAh"]),
]

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")


def _tokens(s: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(s.lower()):
        if t[0].isdigit():
            t = str(float(t)).rstrip("0").rstrip(".")  # "08" == "8"
        out.append(t)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _score(query: str, qtok: set[str], name: str) -> float:
    if _norm(name) == _norm(query):
        return 2.0
    ntok = _tokens(name)
    nums = {t for t in ntok if t[0].isdigit()}
    qnums = {t for t in qtok if t[0].isdigit()}
    if not nums <= qnums:
        return 0.0
    words = [t for t in ntok if not t[0].isdigit()]
    if not words:
        return 1.0 if nums else 0.0
    hit = sum(1 for w in words if w in qtok) / len(words)
    # a catalog name with numbers the query lacks already failed; penalise query numbers the name lacks
    return hit - 0.05 * len(qnums - nums)


def lookup(part_number: Optional[str]) -> Optional[dict]:
    """Fuzzy catalog lookup. Returns a deep copy of the entry, or None if nothing matches well."""
    if not part_number:
        return None
    qtok = set(_tokens(part_number))
    best, best_s = None, 0.0
    for e in CATALOG:
        s = max(_score(part_number, qtok, n) for n in [e["name"], *e["aliases"]])
        if s > best_s:
            best, best_s = e, s
    if best is None or best_s < 0.6:
        return None
    return copy.deepcopy(best)
