"""Generator for the endurance drone. Usage: python make_design.py v01   (parameter sets are in VERSIONS below).
All lengths in metres inside this script; files are written in mm. x aft from the nose, y right, z up."""
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

from foils import piece, cat

BASE = dict(
    # wing
    span=3.0, c_root=0.22, c_tip=0.14, wing_code="2412", x_wing=0.35,
    ail_in=0.55, ail_out=1.35, ail_frac=0.75,
    spar_od=16.0, spar_id=14.0, spar_half=1.40, spar_x_frac=0.25,
    # fuselage / boom
    fus_len=1.0, fus_d=0.14, boom_od=16.0, boom_id=14.0, boom_end=1.42,
    # tail
    ht_span=0.55, ht_c=0.11, x_ht=1.30, el_frac=0.70,
    vt_h=0.30, vt_c_root=0.17, vt_c_tip=0.13, x_vt=1.28, ru_frac=0.70,
    # propulsion
    motor="SunnySky X2216 880KV", prop="APC 10x7E", esc="Generic BLHeli 30A ESC",
    battery="4S2P Molicel P42A Li-ion", batt_x=0.12, batt_len=0.20, batt_w=0.05, batt_h=0.05,
    camera="RunCam Split 4", camera_x=0.06,
    cruise_kmh=55,
    mat=dict(name="LW-PLA", infill=0.0, walls=2, orientation="vase"),
    servo="Emax ES08MA II", z_wing=0.0,
)

VERSIONS = {
    "v01": {},
    "v02": {"prop": "APC 9x6E"},
    "v03": {"prop": "APC 9x6E", "cruise_kmh": 40},
    "v04": {"prop": "APC 9x6E", "cruise_kmh": 40, "battery": "6S2P Molicel P42A Li-ion", "motor": "BrotherHobby Avenger 2816 620KV", "batt_w": 0.055, "batt_h": 0.055},
    "v06": {"prop": "APC 9x6E", "cruise_kmh": 40, "batt_x": 0.15, "batt_len": 0.19, "batt_w": 0.055, "batt_h": 0.055,
            "spar_od": 12.0, "spar_id": 10.0,
            "batt_specs": {"name": "4S3P P42A custom pack", "cells_s": 4, "cells_p": 3, "capacity_mah": 12600, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 880, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 181.4}},
    "v07": {"prop": "APC 9x6E", "cruise_kmh": 40, "batt_x": 0.12, "batt_len": 0.19, "batt_w": 0.055, "batt_h": 0.055,
            "spar_od": 12.0, "spar_id": 10.0, "x_wing": 0.31,
            "batt_specs": {"name": "4S4P P42A custom pack", "cells_s": 4, "cells_p": 4, "capacity_mah": 16800, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 1173, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 241.9}},
    "v08": {"prop": "APC 9x6E", "cruise_kmh": 40, "batt_x": 0.08, "batt_len": 0.21, "batt_w": 0.058, "batt_h": 0.058,
            "spar_od": 12.0, "spar_id": 10.0, "x_wing": 0.30, "ht_span": 0.40,
            "batt_specs": {"name": "4S5P P42A custom pack", "cells_s": 4, "cells_p": 5, "capacity_mah": 21000, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 1467, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 302.4}},
    "v09": {"prop": "APC 9x6E", "cruise_kmh": 40, "batt_x": 0.135, "batt_len": 0.21, "batt_w": 0.058, "batt_h": 0.058,
            "spar_od": 12.0, "spar_id": 10.0, "x_wing": 0.30, "ht_span": 0.40, "z_wing": 0.075,
            "batt_specs": {"name": "4S5P P42A custom pack", "cells_s": 4, "cells_p": 5, "capacity_mah": 21000, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 1467, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 302.4}},
    "v10": {"prop": "APC 9x6E", "cruise_kmh": 40, "batt_x": 0.135, "batt_len": 0.21, "batt_w": 0.058, "batt_h": 0.058,
            "spar_od": 14.0, "spar_id": 12.0, "spar_half": 1.70, "x_wing": 0.30, "ht_span": 0.48, "z_wing": 0.075,
            "span": 3.6, "ail_in": 0.65, "ail_out": 1.65, "vt_h": 0.36,
            "batt_specs": {"name": "4S5P P42A custom pack", "cells_s": 4, "cells_p": 5, "capacity_mah": 21000, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 1467, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 302.4}},
    "v11": {"prop": "APC 9x6E", "cruise_kmh": 40, "batt_x": 0.145, "batt_len": 0.22, "batt_w": 0.062, "batt_h": 0.062,
            "spar_od": 14.0, "spar_id": 12.0, "spar_half": 1.70, "x_wing": 0.30, "ht_span": 0.48, "z_wing": 0.075,
            "span": 3.6, "ail_in": 0.65, "ail_out": 1.65, "vt_h": 0.36,
            "batt_specs": {"name": "4S6P P42A custom pack", "cells_s": 4, "cells_p": 6, "capacity_mah": 25200, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 1760, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 362.9}},
    "v12": {"prop": "APC 9x6E", "cruise_kmh": 42, "batt_x": 0.18, "batt_len": 0.26, "batt_w": 0.07, "batt_h": 0.07, "fus_d": 0.16,
            "spar_od": 14.0, "spar_id": 12.0, "spar_half": 1.70, "x_wing": 0.30, "ht_span": 0.48, "z_wing": 0.085,
            "span": 3.6, "ail_in": 0.65, "ail_out": 1.65, "vt_h": 0.36,
            "batt_specs": {"name": "4S8P P42A custom pack", "cells_s": 4, "cells_p": 8, "capacity_mah": 33600, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 2347, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 483.8}},
    "v13": {"prop": "APC 9x6E", "cruise_kmh": 42, "batt_x": 0.15, "batt_len": 0.26, "batt_w": 0.07, "batt_h": 0.07, "fus_d": 0.16,
            "spar_od": 16.0, "spar_id": 14.0, "spar_half": 1.70, "x_wing": 0.30, "ht_span": 0.54, "z_wing": 0.085,
            "span": 3.6, "c_root": 0.25, "c_tip": 0.16, "ail_in": 0.65, "ail_out": 1.65, "vt_h": 0.40,
            "batt_specs": {"name": "4S8P P42A custom pack", "cells_s": 4, "cells_p": 8, "capacity_mah": 33600, "c_rating": 10.7,
                           "ir_mohm_per_cell": 15, "mass_g": 2347, "chemistry": "liion", "nominal_v": 14.4, "energy_wh": 483.8}},
    "v05": {"prop": "APC 9x6E", "cruise_kmh": 40, "battery": "6S2P Molicel P42A Li-ion", "motor": "BrotherHobby Avenger 2816 620KV", "batt_w": 0.055, "batt_h": 0.055, "batt_x": 0.20, "batt_len": 0.18},
}


def build(outdir, P):
    out = Path(outdir)
    (out / "cad").mkdir(parents=True, exist_ok=True)
    half = P["span"] / 2
    cr, ct = P["c_root"], P["c_tip"]
    chord = lambda y: cr + (ct - cr) * abs(y) / half
    xw = P["x_wing"]
    code = P["wing_code"]
    mesh = {}

    def wp(y0, y1, f0=0.0, f1=1.0):
        return piece(1, y0, y1, chord(y0), chord(y1), xw, xw, P["z_wing"], P["z_wing"], code, f0, f1)

    ai, ao, af = P["ail_in"], P["ail_out"], P["ail_frac"]
    mesh["wing"] = cat([wp(-half, -ao), wp(-ao, -ai, 0, af), wp(-ai, ai), wp(ai, ao, 0, af), wp(ao, half)])
    mesh["aileron_L"] = wp(-ao, -ai, af, 1.0)
    mesh["aileron_R"] = wp(ai, ao, af, 1.0)

    # horizontal tail
    hh = P["ht_span"] / 2
    xh, ch, ef = P["x_ht"], P["ht_c"], P["el_frac"]
    mesh["htail"] = piece(1, -hh, hh, ch, ch, xh, xh, 0, 0, "0010", 0, ef)
    mesh["elevator"] = piece(1, -hh, hh, ch, ch, xh, xh, 0, 0, "0010", ef, 1.0)
    # vertical tail (fin from z=0 up)
    xv, vh, vcr, vct, rf = P["x_vt"], P["vt_h"], P["vt_c_root"], P["vt_c_tip"], P["ru_frac"]
    mesh["vtail"] = piece(2, 0.0, vh, vcr, vct, xv, xv, 0, 0, "0010", 0, rf)
    mesh["rudder"] = piece(2, 0.0, vh, vcr, vct, xv, xv, 0, 0, "0010", rf, 1.0)

    # fuselage, boom, spar
    from dronebench.geometry import make_fuselage, make_tube
    mesh["fuselage"] = make_fuselage(P["fus_len"], P["fus_d"])
    b0 = P["fus_len"] - 0.15
    mesh["boom"] = make_tube(P["boom_od"] / 1000, P["boom_id"] / 1000, P["boom_end"] - b0, "x",
                             ((b0 + P["boom_end"]) / 2, 0, 0))
    sx = xw + P["spar_x_frac"] * cr
    mesh["spar"] = make_tube(P["spar_od"] / 1000, P["spar_id"] / 1000, 2 * P["spar_half"], "y", (sx, 0, P["z_wing"]))

    # battery block, motor, prop
    bx, bl = P["batt_x"], P["batt_len"]
    mesh["battery"] = trimesh.creation.box((bl, P["batt_w"], P["batt_h"]))
    mesh["battery"].apply_translation((bx + bl / 2, 0, 0))
    mm = trimesh.creation.cylinder(radius=0.014, height=0.034, sections=32)
    mm.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mm.apply_translation((-0.012, 0, 0))
    mesh["motor"] = mm
    dprop = {"APC 9x6E": 9, "APC 10x7E": 10, "APC 11x8E": 11, "APC 12x8E": 12, "APC 13x8E": 13}[P["prop"]]
    pr = trimesh.creation.cylinder(radius=dprop * 0.0254 / 2, height=0.004, sections=48)
    pr.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    pr.apply_translation((-0.04, 0, 0))
    mesh["prop"] = pr

    for name, m in mesh.items():
        m = m.copy()
        m.apply_scale(1000.0)
        m.export(out / "cad" / f"{name}.stl")

    # ---- BOM
    mat = dict(P["mat"])
    parts = []

    def add(**kw):
        parts.append({k: v for k, v in kw.items() if v is not None})

    add(id="wing", type="wing", role="wing", cad="wing.stl", material=mat, specs={"fits": ["fuselage"]})
    add(id="aileron_L", type="surface", role="aileron", cad="aileron_L.stl", material=mat)
    add(id="aileron_R", type="surface", role="aileron", cad="aileron_R.stl", material=mat)
    add(id="htail", type="tail", role="htail", cad="htail.stl", material=mat, specs={"fits": ["vtail", "boom"]})
    add(id="elevator", type="surface", role="elevator", cad="elevator.stl", material=mat, specs={"fits": ["rudder", "vtail"]})
    add(id="vtail", type="tail", role="vtail", cad="vtail.stl", material=mat, specs={"fits": ["boom"]})
    add(id="rudder", type="surface", role="rudder", cad="rudder.stl", material=mat)
    add(id="fuselage", type="fuselage", role="fuselage", cad="fuselage.stl", material=mat)
    add(id="boom", type="spar", role="tail_boom", cad="boom.stl", material={"name": "CF tube"},
        specs={"outer_d_mm": P["boom_od"], "inner_d_mm": P["boom_id"]}, connects=["fuselage", "htail", "vtail"])
    add(id="spar", type="spar", role="main_spar", cad="spar.stl", material={"name": "CF tube"},
        specs={"outer_d_mm": P["spar_od"], "inner_d_mm": P["spar_id"]}, connects=["wing", "fuselage"])
    bspec = {"fits": ["fuselage"]}
    bpn = P["battery"]
    if P.get("batt_specs"):
        bspec.update(P["batt_specs"])
        bpn = P["batt_specs"].get("name", "custom pack")
    add(id="battery", type="battery", cad="battery.stl", part_number=bpn, connects=["fuselage"], specs=bspec)
    add(id="motor", type="motor", cad="motor.stl", part_number=P["motor"], connects=["fuselage", "prop", "esc"],
        specs={"fits": ["fuselage"]})
    add(id="prop", type="prop", cad="prop.stl", part_number=P["prop"], connects=["motor"])
    add(id="esc", type="esc", part_number=P["esc"], connects=["battery", "motor", "w_batt", "w_motor"],
        specs={"cg_mm": [bx * 1000 + 220, 0, 0]})
    add(id="camera", type="camera", part_number=P["camera"], connects=["fuselage"],
        specs={"cg_mm": [P["camera_x"] * 1000, 0, -20]})
    add(id="fc", type="fc", part_number="Matek F405-WTE", connects=["fuselage"], specs={"cg_mm": [480, 0, 0]})
    add(id="rx", type="rx", part_number="ExpressLRS EP2 receiver", connects=["fc", "w_av"], specs={"cg_mm": [520, 0, 0]})
    add(id="gps", type="gps", part_number="Matek M10Q-5883", connects=["fc", "w_av"], specs={"cg_mm": [560, 0, 40]})
    sy = (ai + ao) / 2
    sxc = (xw + 0.6 * chord(sy)) * 1000
    servos = [("servo_ail_L", "aileron_L", [sxc, -sy * 1000, P["z_wing"] * 1000]), ("servo_ail_R", "aileron_R", [sxc, sy * 1000, P["z_wing"] * 1000]),
              ("servo_ele", "elevator", [(P["x_ht"] - 0.15) * 1000, 0, 0]),
              ("servo_rud", "rudder", [(P["x_vt"] - 0.15) * 1000, 0, 0])]
    for sid, surf, cg in servos:
        add(id=sid, type="servo", part_number=P["servo"], connects=[surf, "w_" + sid], specs={"cg_mm": cg})
        add(id="w_" + sid, type="wire", material={"name": "Silicone wire"}, connects=[sid, "fc"], specs={"mass_g": 9, "cg_mm": cg})
        add(id="hinge_" + surf, type="hinge", parent=surf, connects=[surf, "wing" if "ail" in surf else "htail" if surf == "elevator" else "vtail"],
            specs={"mass_g": 1.0, "cg_mm": cg})
        add(id="link_" + surf, type="linkage", connects=[sid, surf], specs={"mass_g": 3.0, "cg_mm": cg})
    add(id="w_batt", type="wire", material={"name": "Silicone wire"}, connects=["battery", "esc"], specs={"mass_g": 15, "cg_mm": [bx * 1000 + 200, 0, 0]})
    add(id="w_motor", type="wire", material={"name": "Silicone wire"}, connects=["esc", "motor"], specs={"mass_g": 8, "cg_mm": [bx * 1000 + 100, 0, 0]})
    add(id="w_av", type="wire", material={"name": "Silicone wire"}, connects=["fc", "rx", "gps", "esc"], specs={"mass_g": 8, "cg_mm": [500, 0, 0]})

    (out / "bom.yaml").write_text(yaml.safe_dump({"parts": parts}, sort_keys=False))
    mission = {"preset": "endurance", "payload_ids": ["camera"], "cruise_kmh": P["cruise_kmh"], "max_mass_kg": 7.0,
               "frame": {"units": "mm", "forward": "-x", "up": "+z"}}
    (out / "mission.yaml").write_text(yaml.safe_dump(mission, sort_keys=False))
    (out / "params.yaml").write_text(yaml.safe_dump({k: v for k, v in P.items()}, sort_keys=False))


if __name__ == "__main__":
    v = sys.argv[1]
    out = Path("designs") / v
    if out.exists():
        sys.exit(f"{out} exists; never overwrite a version")
    P = dict(BASE)
    P.update(VERSIONS[v])
    build(out, P)
    print("wrote", out)
