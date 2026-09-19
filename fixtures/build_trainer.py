"""Build the 1.2 m trainer fixtures: fixtures/trainer, fixtures/bad_tail and fixtures/step_mix.

    .venv/bin/python fixtures/build_trainer.py [trainer|bad_tail|step_mix ...]

trainer   high-wing 1.2 m trainer, NACA 2412, SunnySky X2216 1100KV + APC 9x6E, 3S 2200 LiPo, 4 servos.
bad_tail  same airframe with tail areas x 0.4 and the battery pushed aft: must fail static_margin / tail_volume.
step_mix  trainer whose four servos are STEP files (FreeCAD-library SG-90 model, CC-BY-3.0), placed by
          rewriting the STEP assembly placement; everything else is STL.
"""
from __future__ import annotations

import math
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import meshkit as mk  # noqa: E402
from build_falcon_v2 import Planform, mirror, mm3  # noqa: E402
from dronebench.components import lookup  # noqa: E402

MM = 1e-3
STEP_URL = ("https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/be5e71491051cef75da46831b6fd77c6313cb497/"
            "Electrical%20Parts/Servos/SG-90/Servo-sg90.step")
STEP_SRC = HERE / "step_mix" / "source" / "Servo-sg90.step"
# SG-90 STEP native frame (mm): width along x (+-5.9), length along y (-4.7..27.7), height along z (0..29.9)
STEP_CENTER_MM = np.array([0.0, 11.5, 14.95])

PARAMS = {
    "wing": {"span_mm": 1200, "root_chord_mm": 200, "tip_chord_mm": 200, "break_y_mm": 599.0, "sweep_le_deg": 0.0,
             "dihedral_deg": 2.0, "root_airfoil": "2412", "tip_airfoil": "2412", "x_le_root_mm": 225,
             "inner_panel_mm": 300, "aileron_chord_frac": 0.25,
             "material": {"name": "LW-PLA", "infill": 0.0, "walls": 2, "orientation": "vase"}},
    "tail": {"arm_mm": 560, "vh": 0.45, "vv": 0.035, "htail_ar": 3.6, "vtail_height_mm": 140, "airfoil": "0009",
             "elevator_chord_frac": 0.35, "rudder_chord_frac": 0.35, "scale": 1.0},
    "fuselage_wall_mm": 2.0,
    "motor": "SunnySky X2216 1100KV", "prop": "APC 9x6E", "esc": "Generic BLHeli 30A ESC",
    "battery": "3S 2200mAh 30C LiPo", "battery_size_mm": [106, 34, 26],
    "servo_aileron": "TowerPro SG90", "servo_tail": "TowerPro MG90S",
    "spar": {"od_mm": 8, "id_mm": 6, "half_length_mm": 450},
    "target_cg_frac_mac": 0.30,
    "servo_step": False,
}

VARIANTS = {
    "trainer": {},
    "bad_tail": {"tail": {"scale": 0.4}, "target_cg_frac_mac": 0.46},
    "step_mix": {"servo_step": True, "servo_tail": "TowerPro SG90"},
}


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


# ---------------------------------------------------------------- STEP helpers
def fetch_step() -> Path | None:
    if STEP_SRC.exists():
        return STEP_SRC
    STEP_SRC.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(STEP_URL, timeout=60) as r:
            data = r.read()
        if len(data) > 5_000_000 or b"ISO-10303-21" not in data[:100]:
            raise ValueError("unexpected STEP download")
        STEP_SRC.write_bytes(data)
        return STEP_SRC
    except Exception as e:  # network down: fall back to a local copy if the repo has one
        local = HERE.parent / "tests" / "data" / "servo_sg90.step"
        if local.exists():
            shutil.copy(local, STEP_SRC)
            return STEP_SRC
        print(f"warning: could not fetch STEP servo ({e}); step_mix skipped")
        return None


def place_step(src: Path, dst: Path, center_m, axis_z, ref_x):
    """Copy a single-part STEP assembly and move its part by rewriting the child placement (#16/#17/#18 in the
    FreeCAD export): the part's local z maps to `axis_z`, local x to `ref_x`, its centre lands on `center_m`."""
    ez, ex = np.asarray(axis_z, float), np.asarray(ref_x, float)
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])
    loc = np.asarray(center_m, float) / MM - R @ STEP_CENTER_MM
    txt = src.read_text()

    def fmt(v):
        return "(" + ",".join(f"{c:.6f}" for c in v) + ")"
    txt, n1 = re.subn(r"#16 = CARTESIAN_POINT\('',\([^)]*\)\);", f"#16 = CARTESIAN_POINT('',{fmt(loc)});", txt, 1)
    txt, n2 = re.subn(r"#17 = DIRECTION\('',\([^)]*\)\);", f"#17 = DIRECTION('',{fmt(ez)});", txt, 1)
    txt, n3 = re.subn(r"#18 = DIRECTION\('',\([^)]*\)\);", f"#18 = DIRECTION('',{fmt(ex)});", txt, 1)
    if (n1, n2, n3) != (1, 1, 1):
        raise ValueError("STEP placement entities not found")
    dst.write_text(txt)


# ---------------------------------------------------------------- build
def build(name: str, out: Path, verbose=True):
    P = _merge(PARAMS, VARIANTS[name])
    W, T = P["wing"], P["tail"]
    A = mk.Assembly()
    plan0 = Planform(W, 0.0)
    S, mac, _, x_le_mac = plan0.area_mac()
    x_ac = x_le_mac + 0.25 * mac
    b = W["span_mm"] * MM
    lh = T["arm_mm"] * MM
    Sh = T["vh"] * S * mac / lh * T["scale"]
    Sv = T["vv"] * S * b / lh * T["scale"]
    bh = math.sqrt(Sh * T["htail_ar"])
    ch = Sh / bh
    hv = T["vtail_height_mm"] * MM * math.sqrt(T["scale"])
    crv = 2 * Sv / (hv * 1.7)
    ctv = 0.7 * crv
    x_le_h = x_ac + lh - 0.25 * ch
    tail_end = x_le_h + ch + 0.008
    x_le = W["x_le_root_mm"] * MM
    cr = W["root_chord_mm"] * MM

    hull = mk.Hull([(0.000, 0.060, 0.064, 0.0), (0.120, 0.085, 0.095, -0.005), (0.400, 0.085, 0.095, -0.005),
                    (0.600, 0.052, 0.058, 0.008), (x_le_h - 0.03, 0.026, 0.030, 0.016), (tail_end, 0.026, 0.030, 0.016)],
                   wall=P["fuselage_wall_mm"] * MM)
    fuse_top = hull.top(x_le)
    _, lo = mk.naca4_surfaces(W["root_airfoil"], mk.chord_stations(0, 1, 200))
    z_root = fuse_top - lo[:, 1].min() * cr + 0.0003
    plan = Planform(W, z_root)
    fmat = {"name": "LW-PLA", "infill": 0.0, "walls": 3, "orientation": "flat"}
    A.add("fuse_front", "fuselage", hull.section(0.0, 0.40), role="fuselage", part_number="Trainer fuselage front",
          material=fmat)
    A.add("fuse_rear", "fuselage", hull.section(0.40, tail_end), role="fuselage", part_number="Trainer tail boom",
          material=fmat)

    # wing: inner panels full chord, outer panels with ailerons
    yi = W["inner_panel_mm"] * MM
    ca = W["aileron_chord_frac"]
    for side, sgn in (("L", -1), ("R", 1)):
        st_in, st_out = plan.stations(0, yi), plan.stations(yi, plan.b2)
        if sgn < 0:
            st_in, st_out = mirror(st_in), mirror(st_out)
        A.add(f"wing_{side}1", "wing", mk.loft_surface(st_in), role="wing", part_number=f"Trainer wing inner {side}",
              material=W["material"])
        A.add(f"wing_{side}2", "wing", mk.loft_surface(st_out, 0, 1 - ca), role="wing",
              part_number=f"Trainer wing outer {side}", material=W["material"])
        A.add(f"aileron_{side}", "surface", mk.loft_surface(st_out, 1 - ca, 1), role="aileron", parent=f"wing_{side}2",
              part_number=f"Trainer aileron {side}",
              material={"name": "LW-PLA", "infill": 0.0, "walls": 1, "orientation": "vase"})

    def camber_z(y, frac):
        u, l = mk.naca4_surfaces(plan.airfoil(y), np.array([frac]))
        return plan.z(y) + 0.5 * (u[0, 1] + l[0, 1]) * plan.chord(y)

    sp = P["spar"]
    wing_ids = [f"wing_{s}{i}" for s in "LR" for i in (1, 2)]
    A.add("main_spar", "spar", mk.make_tube(sp["od_mm"] * MM, sp["id_mm"] * MM, 2 * sp["half_length_mm"] * MM,
                                            (x_le + 0.25 * cr, 0, camber_z(0, 0.25) + 0.002), "y"),
          role="main_spar", part_number=f"CF tube {sp['od_mm']}/{sp['id_mm']} mm pultruded", material={"name": "CF tube"},
          specs={"outer_d_mm": sp["od_mm"], "inner_d_mm": sp["id_mm"], "length_mm": 2 * sp["half_length_mm"],
                 "fits": wing_ids})

    # tail
    af = T["airfoil"]
    _, lo_h = mk.naca4_surfaces(af, mk.chord_stations(0, 1, 100))
    z_h = hull.top(x_le_h + 0.5 * ch) - lo_h[:, 1].min() * ch + 0.0003
    ce, crf = T["elevator_chord_frac"], T["rudder_chord_frac"]
    hst = [(-bh / 2, ch, x_le_h, z_h, af), (bh / 2, ch, x_le_h, z_h, af)]
    tmat = {"name": "LW-PLA", "infill": 0.0, "walls": 2, "orientation": "vase"}
    smat = {"name": "LW-PLA", "infill": 0.0, "walls": 1, "orientation": "vase"}
    A.add("hstab", "tail", mk.loft_surface(hst, 0, 1 - ce), role="htail", part_number="Trainer stabilizer",
          material=tmat)
    A.add("elevator", "surface", mk.loft_surface(hst, 1 - ce, 1), role="elevator", parent="hstab",
          part_number="Trainer elevator", material=smat)
    up_h, _ = mk.naca4_surfaces(af, mk.chord_stations(0, 1, 100))
    z_v0 = z_h + up_h[:, 1].max() * ch + 0.0003
    x_le_v = x_le_h + ch - crv
    vst = [(z_v0, crv, x_le_v, 0.0, af), (z_v0 + hv, ctv, x_le_v + (crv - ctv), 0.0, af)]
    A.add("vstab", "tail", mk.loft_surface(vst, 0, 1 - crf, axis="z"), role="vtail", part_number="Trainer fin",
          material=tmat)
    A.add("rudder", "surface", mk.loft_surface(vst, 1 - crf, 1, axis="z"), role="rudder", parent="vstab",
          part_number="Trainer rudder", material=smat)

    # propulsion: firewall, motor, prop
    mot, prop, esc = lookup(P["motor"]), lookup(P["prop"]), lookup(P["esc"])
    zc0 = 0.0
    A.add("firewall", "mount", mk.box((0.003, 0.050, 0.050), (-0.0015, 0, zc0)), part_number="3 mm plywood firewall",
          material={"name": "Plywood"})
    ml, sl, hl = 0.026, 0.012, 0.010
    A.add("motor", "motor", mk.motor((-0.003 - ml / 2, 0, zc0), 0.028, ml, sl), part_number=P["motor"],
          specs=mot["specs"], connects=["prop"], material={"name": "Motor assembly", "sub": mot["sub"]})
    A.add("prop", "prop", mk.prop((-0.003 - ml - sl - hl / 2 + 0.0002, 0, zc0), prop["specs"]["diam_in"] * 0.0254,
                                  prop["specs"]["pitch_in"] * 0.0254, hub_d=0.016, hub_len=hl),
          part_number=P["prop"], specs=prop["specs"])
    A.add("motor_screws", "fastener", group="motor_mount", qty=4, part_number="M3x8 steel screw",
          specs={"size": "M3x8", "mass_g": 0.8, "cg_mm": [-2.0, 0.0, 0.0]}, connects=["motor", "firewall"])

    # tray + avionics
    tx0, tx1, tw = 0.020, 0.360, 0.050
    tz = hull.floor(tx0, tx1, tw / 2) + 0.0002
    A.add("tray", "mount", mk.box((tx1 - tx0, tw, 0.002), ((tx0 + tx1) / 2, 0, tz + 0.001)),
          part_number="Trainer equipment tray", material={"name": "Plywood"})
    top = tz + 0.002

    def on_tray(id_, typ, size, x, y=0.0, **kw):
        A.add(id_, typ, mk.box(size, (x, y, top + size[2] / 2 + 0.0001)), **kw)

    esc_x, fc_x, gps_x, servo_x = 0.205, 0.250, 0.305, 0.335
    on_tray("esc", "esc", (0.050, 0.025, 0.010), esc_x, part_number=P["esc"], specs=esc["specs"], connects=["motor"])
    on_tray("fc", "fc", (0.036, 0.036, 0.010), fc_x, part_number="Matek F405-WTE",
            specs=lookup("Matek F405-WTE")["specs"], connects=["esc"])
    on_tray("rx", "rx", (0.018, 0.012, 0.004), fc_x + 0.035, part_number="ExpressLRS EP2 receiver",
            specs=lookup("ELRS EP2")["specs"], connects=["fc"])
    on_tray("gps", "gps", (0.020, 0.020, 0.008), gps_x, part_number="Matek M10Q-5883",
            specs=lookup("Matek M10Q-5883")["specs"], connects=["fc"])

    # servos
    servo_step = P["servo_step"]
    step_src = fetch_step() if servo_step else None
    if servo_step and step_src is None:
        return None
    steps = {}

    def add_servo(sid, pn, center, orient, surf, fits):
        s = lookup(pn)
        d = [v * MM for v in s["specs"]["dims_mm"]]
        spec = dict(s["specs"], fits=fits) if fits else dict(s["specs"])
        if servo_step:
            axes = {"flat": ((1, 0, 0), (0, 0, -1)), "z": ((0, 0, 1), (0, 1, 0))}[orient]
            steps[sid] = (center, *axes)
            A.cg_hint[sid] = np.asarray(center, float)
            A.add(sid, "servo", None, part_number=pn, specs=spec, connects=[surf, "fc"])
            A.get(sid)["cad"] = f"{sid}.step"
        else:
            A.add(sid, "servo", mk.servo(center, d, orient=orient), part_number=pn, specs=spec, connects=[surf, "fc"])

    for side, sgn in (("L", -1), ("R", 1)):
        y = sgn * (yi + 0.10)
        xs = x_le + (1 - ca) * cr - 0.018
        add_servo(f"servo_aileron_{side}", P["servo_aileron"], (xs, y, camber_z(y, (xs - x_le) / cr)), "flat",
                  f"aileron_{side}", [f"wing_{side}2"])
        hx = x_le + (1 - ca) * cr + 0.012
        zl = plan.z(y) + mk.naca4_surfaces(plan.airfoil(y), np.array([(hx - x_le) / cr]))[1][0, 1] * cr
        A.add(f"horn_aileron_{side}", "horn", mk.box((0.010, 0.002, 0.012), (hx, y, zl - 0.005)),
              parent=f"aileron_{side}", part_number="control horn (nylon)", material={"name": "Nylon PA66"},
              specs={"fits": [f"aileron_{side}"]})
        A.add(f"link_aileron_{side}", "linkage", mk.rod((xs + 0.0125, y + 0.002, zl - 0.004), (hx, y + 0.002, zl - 0.008),
                                                        0.0012),
              part_number="1.2 mm steel pushrod", material={"name": "Steel"},
              specs={"fits": [f"wing_{side}2", f"aileron_{side}", f"horn_aileron_{side}", f"servo_aileron_{side}"]},
              connects=[f"servo_aileron_{side}", f"aileron_{side}"])
        u, l = mk.naca4_surfaces(plan.airfoil(y), np.array([1 - ca]))
        p0 = [x_le + (1 - ca) * cr, sgn * (yi + 0.005), plan.z(yi + 0.005) + 0.5 * (u[0, 1] + l[0, 1]) * cr]
        p1 = [x_le + (1 - ca) * cr, sgn * (plan.b2 - 0.005), plan.z(plan.b2 - 0.005) + 0.5 * (u[0, 1] + l[0, 1]) * cr]
        A.add(f"hinge_aileron_{side}", "hinge", mk.rod(p0, p1, 0.0015), parent=f"aileron_{side}",
              part_number="CA hinge strip + 1.5 mm steel pin", material={"name": "Steel"},
              specs={"fits": [f"aileron_{side}", f"wing_{side}2"]})

    sd = lookup(P["servo_tail"])["specs"]["dims_mm"]
    for sid, surf, y in (("servo_elevator", "elevator", -0.013), ("servo_rudder", "rudder", 0.013)):
        c = (servo_x, y, top + sd[2] * MM / 2 + 0.0001)
        add_servo(sid, P["servo_tail"], c, "z", surf, [])
    # tail horns + long pushrods through the boom
    hx = x_le_h + (1 - ce) * ch + 0.010
    A.add("horn_elevator", "horn", mk.box((0.010, 0.002, 0.012), (hx, 0.02, z_h - 0.004)), parent="elevator",
          part_number="control horn (nylon)", material={"name": "Nylon PA66"}, specs={"fits": ["elevator"]})
    A.add("link_elevator", "linkage", mk.rod((servo_x, -0.013, top + 0.030), (hx, 0.02, z_h - 0.008), 0.0015),
          part_number="1.5 mm steel pushrod in tube", material={"name": "Steel"},
          specs={"fits": ["fuse_front", "fuse_rear", "hstab", "elevator", "horn_elevator", "servo_elevator"]},
          connects=["servo_elevator", "elevator"])
    xr = x_le_v + (1 - crf) * crv + 0.010
    A.add("horn_rudder", "horn", mk.box((0.010, 0.012, 0.002), (xr, 0.008, z_v0 + 0.015)), parent="rudder",
          part_number="control horn (nylon)", material={"name": "Nylon PA66"}, specs={"fits": ["rudder"]})
    A.add("link_rudder", "linkage", mk.rod((servo_x, 0.013, top + 0.030), (xr, 0.010, z_v0 + 0.012), 0.0015),
          part_number="1.5 mm steel pushrod in tube", material={"name": "Steel"},
          specs={"fits": ["fuse_front", "fuse_rear", "hstab", "vstab", "rudder", "horn_rudder", "servo_rudder"]},
          connects=["servo_rudder", "rudder"])
    A.add("hinge_elevator", "hinge", mk.rod((x_le_h + (1 - ce) * ch, -bh / 2 + 0.005, z_h), (x_le_h + (1 - ce) * ch,
                                                                                          bh / 2 - 0.005, z_h), 0.0015),
          parent="elevator", part_number="CA hinge strip + 1.5 mm steel pin", material={"name": "Steel"},
          specs={"fits": ["elevator", "hstab"]})
    A.add("hinge_rudder", "hinge", mk.rod((x_le_v + (1 - crf) * crv, 0, z_v0 + 0.004),
                                          (x_le_v + (crv - ctv) + (1 - crf) * ctv, 0, z_v0 + hv - 0.004), 0.0015),
          parent="rudder", part_number="CA hinge strip + 1.5 mm steel pin", material={"name": "Steel"},
          specs={"fits": ["rudder", "vstab"]})
    fc = A.get("fc")
    fc["connects"] = fc["connects"] + ["servo_aileron_L", "servo_aileron_R", "servo_elevator", "servo_rudder"]

    A.add("wing_bolts", "fastener", group="wing_mount", qty=2, part_number="M5x30 nylon wing bolt",
          material={"name": "Nylon PA66"}, specs={"size": "M5x30", "mass_g": 1.2, "cg_mm": mm3((x_le + 0.15, 0, fuse_top))},
          connects=["wing_L1", "wing_R1", "fuse_front"])
    A.add("servo_screws", "fastener", group="servo_mount", qty=8, part_number="M2x8 steel screw",
          specs={"size": "M2x8", "mass_g": 0.25, "cg_mm": mm3((0.30, 0, top))},
          connects=["servo_aileron_L", "servo_aileron_R", "servo_elevator", "servo_rudder"])
    A.add("wiring", "wire", part_number="silicone wiring + XT60", specs={"mass_g": 35, "cg_mm": mm3((0.15, 0, top + 0.01))},
          connects=["battery", "esc"])

    # battery on the tray between ESC and FC, placed for the target CG
    bat = lookup(P["battery"])
    bsz = [v * MM for v in P["battery_size_mm"]]
    m0, cg0, _ = A.mass_cg()
    x_t = x_le_mac + P["target_cg_frac_mac"] * mac
    bx = (x_t * (m0 + bat["mass_g"]) - cg0[0] * m0) / bat["mass_g"]
    lo_x, hi_x = tx0 + 0.002 + bsz[0] / 2, esc_x - 0.025 - 0.002 - bsz[0] / 2
    bx = round(min(max(bx, lo_x), hi_x), 3)
    bz = top + bsz[2] / 2 + 0.0001
    if name == "bad_tail":   # pack moved aft into the tail boom, resting on the hull floor
        bx = 0.420
        bz = hull.floor(bx - bsz[0] / 2, bx + bsz[0] / 2, bsz[1] / 2) + 0.0002 + bsz[2] / 2
    A.add("battery", "battery", mk.box(bsz, (bx, 0, bz)), part_number=P["battery"], specs=bat["specs"],
          connects=["esc", "fc"])

    total, cg, rows = A.mass_cg()
    note = (f"S={S:.4f} m2 MAC={mac * 1000:.0f} mm x_ac={x_ac * 1000:.1f} mm Sh={Sh:.4f} m2 Sv={Sv:.4f} m2 "
            f"CG x={cg[0] * 1000:.1f} mm ({(cg[0] - x_le_mac) / mac:.3f} MAC)")
    header = (f"{name}: generated by fixtures/build_trainer.py. Frame: mm, x aft, y right, z up.\n{note}\n"
              f"AUW estimate (dronebench.materials.part_mass_g): {total:.0f} g\n"
              "specs.fits lists parts this part is allowed to sit inside (declared fits for the clearance check).")
    mission = {"preset": "endurance", "payload_ids": [], "frame": {"units": "mm", "forward": "-x", "up": "+z"}}
    A.write(out, mission, header)
    for sid, (c, ez, ex) in steps.items():
        place_step(step_src, out / "cad" / f"{sid}.step", c, ez, ex)
    if verbose:
        print(f"{name}: {note}\n  AUW {total:.0f} g, {len(A.parts)} parts")
    return {"S": S, "mac": mac, "x_ac": x_ac, "mass_g": total, "cg": cg, "Sh": Sh, "Sv": Sv, "rows": rows}


def main(argv=None):
    names = (argv if argv is not None else sys.argv[1:]) or list(VARIANTS)
    for n in names:
        build(n, HERE / n)


if __name__ == "__main__":
    main()
