"""Download a real open fixed-wing design and turn it into fixtures/third_party.

    .venv/bin/python fixtures/fetch_third_party.py [--cache DIR]

Source: OpenGeoDrone (https://github.com/enacuavlab/OpenGeoDrone, GPL-3.0), a parametric vase-mode
VTOL flying wing (4 hover motors on two arms + 1 pusher, 2 elevons). The repo's per-part STLs are
~190 MB, so we fetch the Bambu Studio print-plate 3MF files instead (~8.4 MB). Those hold the same
part meshes (decimated, re-centred on their own bounding box, plate transforms separate). We drop
the plate transforms and re-assemble the parts in the OpenSCAD frame by stacking them spanwise from
the OpenGeoDrone.scad parameters and matching airfoil sections at each joint (see SOURCE.md).

Output frame = OpenSCAD frame, left untouched on purpose so ingest must convert it:
units mm, x aft (forward "-x"), y up ("+y"), z = LEFT span.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meshkit as mk  # noqa: E402

REPO = "enacuavlab/OpenGeoDrone"
COMMIT = "4c611977679d9378b03034ca3fedfd96186c708e"
PLATES = ["Center_part", "Fuselage_and_Clamps", "Left_Mid_Aileron_part", "Left_Root_part",
          "Motor_Arms_and_sides", "Right_Mid_Aileron_part", "Right_Root_part"]
NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
MAX_BYTES = 60_000_000

# OpenGeoDrone.scad parameters used for the spanwise stack (mm)
CENTER_WIDTH = 90          # center_width
ROOT_LEN = 205             # wing_root_mm - motor_arm_to_wing_hull
ARM_WIDTH = 38             # motor arm body width (bbox of Motor_arm_* along z)
MID_LEN = 247              # Mid_Aileron_part span (bbox)


def fetch(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    out, total = {}, 0
    for p in PLATES:
        f = cache / f"X1C_{p}.3mf"
        if not f.exists():
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/3mf_print_part/X1C_{p}.3mf"
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            f.write_bytes(data)
        total += f.stat().st_size
        if total > MAX_BYTES:
            raise RuntimeError("download budget exceeded")
        out[p] = f
    return out


def plate_objects(path: Path) -> dict[str, list[trimesh.Trimesh]]:
    """{part name (from Bambu model_settings): [meshes in object coordinates, plate transform dropped]}"""
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    names = dict(re.findall(r'<object id="(\d+)">\s*<metadata key="name" value="([^"]*)"',
                            z.read("Metadata/model_settings.config").decode()))
    meshes = {}
    for o in root.iter(NS + "object"):
        m = o.find(NS + "mesh")
        if m is not None:
            v = np.array([[float(x.get(k)) for k in "xyz"] for x in m.find(NS + "vertices")])
            t = np.array([[int(x.get(k)) for k in ("v1", "v2", "v3")] for x in m.find(NS + "triangles")])
            meshes[o.get("id")] = trimesh.Trimesh(v, t, process=True)
    out: dict[str, list] = {}
    for o in root.iter(NS + "object"):
        c = o.find(NS + "components")
        if c is not None:
            name = names[o.get("id")].replace(".stl", "")
            out.setdefault(name, []).extend(meshes[cc.get("objectid")].copy() for cc in c)
    return out


def section(m, z, tol=1.5):
    v = m.vertices
    s = v[np.abs(v[:, 2] - z) < tol]
    return dict(le=s[:, 0].min(), te=s[:, 0].max(), ymin=s[:, 1].min(), ymax=s[:, 1].max())


def mid(s):
    return np.array([(s["le"] + s["te"]) / 2, (s["ymin"] + s["ymax"]) / 2])


def assemble(parts: dict[str, trimesh.Trimesh]) -> dict[str, np.ndarray]:
    """Translation (mm, OpenSCAD frame) for every part. Left side = +z."""
    T = {}
    c = parts["Center_part"]
    T["Center_part"] = np.zeros(3)
    cz = c.bounds[1, 2]
    c_face = mid(section(c, cz - 1))
    # root: inner end (local z min) meets the center part side face; match section mid-points
    r = parts["Left_Root_part"]
    r_in = mid(section(r, r.bounds[0, 2] + 1))
    r_out_s = section(r, r.bounds[1, 2] - 1)
    r_mid_s = section(r, 0.0)
    dxy = c_face - r_in
    z_root = CENTER_WIDTH / 2 + ROOT_LEN / 2
    T["Left_Root_part"] = np.array([dxy[0], dxy[1], z_root])
    # sweep slope of the root panel (LE and TE), used to bridge the motor-arm gap
    zr = r.bounds[1, 2] - 1
    sl_le = (r_out_s["le"] - r_mid_s["le"]) / zr
    sl_te = (r_out_s["te"] - r_mid_s["te"]) / zr
    root_out_g = mid(r_out_s) + dxy
    gap = ARM_WIDTH + 2.0
    m = parts["Left_Mid_Aileron_part"]
    m_in = mid(section(m, m.bounds[0, 2] + 1))
    target = root_out_g + np.array([0.5 * (sl_le + sl_te) * gap, 0.0])
    T["Left_Mid_Aileron_part"] = np.array([*(target - m_in), CENTER_WIDTH / 2 + ROOT_LEN + ARM_WIDTH + MID_LEN / 2])
    # motor arm: centred in the gap, front/back halves meeting at 40 % chord of the local section
    z_arm = CENTER_WIDTH / 2 + ROOT_LEN + ARM_WIDTH / 2
    le_arm = r_out_s["le"] + dxy[0] + sl_le * (ARM_WIDTH / 2 + 1)
    te_arm = r_out_s["te"] + dxy[0] + sl_te * (ARM_WIDTH / 2 + 1)
    j = le_arm + 0.40 * (te_arm - le_arm)
    fr, bk = parts["Left_Motor_arm_front"], parts["Left_Motor_arm_back"]
    y_arm = root_out_g[1]
    T["Left_Motor_arm_front"] = np.array([j - fr.bounds[1, 0], y_arm - fr.bounds.mean(0)[1], z_arm])
    T["Left_Motor_arm_back"] = np.array([j - bk.bounds[0, 0], y_arm - bk.bounds.mean(0)[1], z_arm])
    # servo horn under the elevon root of the mid part
    h = parts["Left_Servo_horn"]
    mz = m.bounds[0, 2] + 30
    ms_ = section(m, mz)
    tm = T["Left_Mid_Aileron_part"]
    T["Left_Servo_horn"] = np.array([ms_["te"] - 20 + tm[0] - h.bounds.mean(0)[0],
                                     ms_["ymin"] + tm[1] - h.bounds[1, 1], mz + tm[2]])
    # right side: mirror of the left placement (right meshes are already mirrored)
    for k in list(T):
        if k.startswith("Left_"):
            t = T[k].copy()
            t[2] = -t[2]
            T["Right_" + k[5:]] = t
    # fuselage pod: tail flush with the center part TE, parts stacked as cut in OpenGeoDrone.scad
    f, bb, up = parts["Fuselage_front_part"], parts["Fuselage_bottom_back_part"], parts["Fuselage_upper_part"]
    lf = f.extents[0]
    total = lf + bb.extents[0] - 12.0                # bottom/back overlaps the front by 12 mm
    tail = c.bounds[1, 0]
    nose = tail - total
    top = f.bounds[1, 1]                              # front part spans the full height, centred on y = 0
    T["Fuselage_front_part"] = np.array([nose - f.bounds[0, 0], 0.0, 0.0])
    # bottom/back: its front end (y range at x min) matches the front part's aft lower lip
    fs = f.vertices[f.vertices[:, 0] > f.bounds[1, 0] - 2]
    bs = bb.vertices[bb.vertices[:, 0] < bb.bounds[0, 0] + 2]
    dy_b = 0.5 * (fs[:, 1].min() + fs[:, 1].max()) - 0.5 * (bs[:, 1].min() + bs[:, 1].max())
    T["Fuselage_bottom_back_part"] = np.array([tail - bb.bounds[1, 0], dy_b, 0.0])
    T["Fuselage_upper_part"] = np.array([tail - up.bounds[1, 0], top - up.bounds[1, 1], 0.0])
    # rear motor plate on the tail face of the bottom/back part
    rm = parts["Rear_motor_part"]
    ts = bb.vertices[bb.vertices[:, 0] > bb.bounds[1, 0] - 2]
    T["Rear_motor_part"] = np.array([tail - rm.bounds[0, 0], 0.5 * (ts[:, 1].min() + ts[:, 1].max()) + dy_b, 0.0])
    return T


def build(out: Path, cache: Path):
    files = fetch(cache)
    parts: dict[str, trimesh.Trimesh] = {}
    clamps = []
    for p, f in files.items():
        for name, ms in plate_objects(f).items():
            if name == "Clamp_fixation_big":
                clamps.extend(ms)
            else:
                parts[name] = ms[0]
    T = assemble(parts)
    placed = {k: parts[k].copy() for k in T}
    for k, t in T.items():
        placed[k].apply_translation(t)

    A = mk.Assembly(scale=1.0)   # meshes already in mm
    frame = {"units": "mm", "forward": "-x", "up": "+y"}
    petg = {"name": "PETG", "infill": 0.15, "walls": 2, "orientation": "flat"}
    aero = {"name": "LW-PLA", "infill": 0.0, "walls": 1, "orientation": "vase"}
    ogd = "OpenGeoDrone "

    def add_mesh(pid, typ, key, ext=".stl", **kw):
        A.add(pid, typ, placed[key], cad_ext=ext, **kw)

    add_mesh("center_part", "fuselage", "Center_part", role="fuselage", part_number=ogd + "Center_part",
             material=petg, notes="center body; carries the spars and wing roots",
             specs={"fits": ["fuselage_front", "fuselage_bottom_back", "fuselage_upper"]})
    add_mesh("fuselage_front", "fuselage", "Fuselage_front_part", role="fuselage", part_number=ogd + "Fuselage_front_part",
             material=petg)
    add_mesh("fuselage_bottom_back", "fuselage", "Fuselage_bottom_back_part", role="fuselage",
             part_number=ogd + "Fuselage_bottom_back_part", material=petg,
             specs={"fits": ["fuselage_front", "fuselage_upper"]})
    add_mesh("fuselage_upper", "fuselage", "Fuselage_upper_part", role="fuselage",
             part_number=ogd + "Fuselage_upper_part", material=petg, specs={"fits": ["fuselage_front"]})
    add_mesh("rear_motor_mount", "mount", "Rear_motor_part", part_number=ogd + "Rear_motor_part", material=petg)
    for side, S in (("L", "Left"), ("R", "Right")):
        add_mesh(f"wing_root_{side}", "wing", f"{S}_Root_part", ext=".3mf", role="wing",
                 part_number=ogd + f"{S}_Root_part", material=aero,
                 notes="PLA Aero (foaming PLA) vase-mode panel with internal grid")
        add_mesh(f"wing_mid_{side}", "wing", f"{S}_Mid_Aileron_part", ext=".3mf", role="wing",
                 part_number=ogd + f"{S}_Mid_Aileron_part", material=aero,
                 notes="mid panel printed together with its elevon (flexure/pin hinge); the elevon is not a "
                       "separate body in the CAD")
        add_mesh(f"motor_arm_front_{side}", "mount", f"{S}_Motor_arm_front", part_number=ogd + f"{S}_Motor_arm_front",
                 material=petg, specs={"fits": [f"wing_root_{side}", f"wing_mid_{side}"]})
        add_mesh(f"motor_arm_back_{side}", "mount", f"{S}_Motor_arm_back", part_number=ogd + f"{S}_Motor_arm_back",
                 material=petg, specs={"fits": [f"wing_root_{side}", f"wing_mid_{side}"]})
        add_mesh(f"servo_horn_{side}", "horn", f"{S}_Servo_horn", part_number=ogd + f"{S}_Servo_horn", material=petg,
                 specs={"fits": [f"wing_mid_{side}"]})

    # six wing clamps at the three joints per side, sunk into the upper skin
    joints = [("center_part", "wing_root"), ("wing_root", "motor_arm_front"), ("motor_arm_front", "wing_mid")]
    zj = [CENTER_WIDTH / 2, CENTER_WIDTH / 2 + ROOT_LEN, CENTER_WIDTH / 2 + ROOT_LEN + ARM_WIDTH]
    k = 0
    for side, sgn in (("L", 1), ("R", -1)):
        for (a, b), z in zip(joints, zj):
            S = "Left" if sgn > 0 else "Right"
            ref = trimesh.util.concatenate([placed[f"{S}_Root_part"], placed[f"{S}_Mid_Aileron_part"]])
            zz = z + 3 if z < CENTER_WIDTH / 2 + ROOT_LEN else z - 3
            zz = z + 5 if z > CENTER_WIDTH / 2 + ROOT_LEN + 1 else zz
            s = section(ref, sgn * zz, tol=2.5)
            cm = clamps[k % len(clamps)].copy()
            cm.apply_translation([(s["le"] + s["te"]) / 2 - cm.bounds.mean(0)[0], s["ymax"] - cm.bounds[1, 1] + 2,
                                  sgn * z - cm.bounds.mean(0)[2]])
            k += 1
            fits = [x if x == "center_part" else f"{x}_{side}" for x in (a, b)]
            A.add(f"clamp_{side}{k}", "fastener", cm, group=f"wing_clamps_{side}", part_number=ogd + "Clamp_fixation_big",
                  material=petg, specs={"fits": fits})

    # ---------------- parts not in the CAD (README "Hardware (not printed)"), cg in the CAD frame (mm)
    arm_f, arm_b = placed["Left_Motor_arm_front"].bounds, placed["Left_Motor_arm_back"].bounds
    z_arm = float(arm_f.mean(0)[2])
    y_top = float(arm_f[1, 1])
    x_hf, x_hb = float(arm_f[0, 0]) + 12, float(arm_b[1, 0]) - 12
    tail = float(placed["Rear_motor_part"].bounds[1, 0])
    y_tail = float(placed["Rear_motor_part"].bounds.mean(0)[1])
    xc = float(placed["Fuselage_front_part"].bounds.mean(0)[0])

    def cg(x, y, z):
        return [round(float(x), 1), round(float(y), 1), round(float(z), 1)]

    hover = {"kv": 1950, "rm_ohm": 0.06, "i0_a": 1.4, "imax_a": 40, "mass_g": 37}
    hover_note = "README: 'F40' hover motors; assumed T-Motor F40 PRO IV 1950KV (approx datasheet values)"
    for i, (x, z, rot) in enumerate([(x_hf, z_arm, "CW"), (x_hb, z_arm, "CCW"), (x_hf, -z_arm, "CCW"),
                                     (x_hb, -z_arm, "CW")], 1):
        A.add(f"hover_motor_{i}", "motor", part_number="T-Motor F40 PRO IV 1950KV", specs={**hover, "cg_mm": cg(x, y_top + 12, z),
              "axis": "vertical"}, connects=[f"hover_prop_{i}"], notes=hover_note)
        A.add(f"hover_prop_{i}", "prop", part_number=f"7x4 {'reverse ' if rot == 'CCW' else ''}prop",
              specs={"diam_in": 7, "pitch_in": 4, "blades": 2, "mass_g": 6, "rotation": rot,
                     "cg_mm": cg(x, y_top + 30, z), "axis": "vertical"}, notes="README: 2x 7x4 + 2x reverse 7x4")
    A.add("esc_4in1", "esc", part_number="4-in-1 ESC 45A (hover)", specs={"imax_a": 45, "mass_g": 16,
          "cg_mm": cg(xc + 30, -10, 0)}, connects=[f"hover_motor_{i}" for i in range(1, 5)],
          notes="README: '1 ESC 4 to 1'; rating assumed")
    A.add("pusher_motor", "motor", part_number="T-Motor AT2312 1150KV",
          specs={"kv": 1150, "rm_ohm": 0.09, "i0_a": 0.8, "imax_a": 25, "mass_g": 60, "cg_mm": cg(tail + 16, y_tail, 0)},
          connects=["pusher_prop"], notes="README: pusher motor 'low kV < 1200'; model assumed")
    A.add("pusher_prop", "prop", part_number="APC 9x6E", specs={"diam_in": 9, "pitch_in": 6, "blades": 2, "mass_g": 15,
                                                            "cg_mm": cg(tail + 40, y_tail, 0)}, notes="README: 9x6 pusher")
    A.add("pusher_esc", "esc", part_number="Generic BLHeli 30A ESC", specs={"imax_a": 30, "mass_g": 25,
          "cg_mm": cg(tail - 40, -15, 0)}, connects=["pusher_motor"])
    A.add("battery", "battery", part_number="4S1P Molicel P42A Li-ion",
          specs={"cells_s": 4, "cells_p": 1, "capacity_mah": 4200, "c_rating": 10.7, "ir_mohm_per_cell": 15,
                 "chemistry": "liion", "mass_g": 300, "cg_mm": cg(0.0, -20, 0)},
          connects=["esc_4in1", "pusher_esc", "fc"], notes="README: '4S 4Ah battery'; pack type assumed")
    A.add("fc", "fc", part_number="Tawaki autopilot (Paparazzi)", specs={"mass_g": 12, "has_remote_id": False,
          "cg_mm": cg(xc + 60, 5, 0)}, connects=["esc_4in1", "pusher_esc", "servo_L", "servo_R"],
          notes="README: 'Flight controller (e.g. Tawaki)'; mass assumed")
    A.add("rx", "rx", part_number="ExpressLRS EP2 receiver", specs={"mass_g": 1, "cg_mm": cg(xc + 90, 10, 10)},
          connects=["fc"], notes="README: 'RC'; model assumed")
    A.add("gps", "gps", part_number="Matek M10Q-5883", specs={"mass_g": 9, "cg_mm": cg(xc + 80, 25, 0)},
          connects=["fc"], notes="README: 'GPS module'; model assumed")
    A.add("telemetry", "telemetry", part_number="SiK 433 MHz telemetry modem", specs={"mass_g": 15,
          "cg_mm": cg(xc + 100, 10, -15)}, connects=["fc"], notes="README: 'Modem to Ground station if needed'")
    A.add("pitot", "sensor", part_number="Matek ASPD-4525", specs={"mass_g": 6,
          "cg_mm": cg(placed["Fuselage_front_part"].bounds[0, 0] + 25, 0, 0)}, connects=["fc"],
          notes="pitot tube through the nose (Create_pitot in OpenGeoDrone.scad)")
    for side, S, sgn in (("L", "Left", 1), ("R", "Right", -1)):
        hb = placed[f"{S}_Servo_horn"].bounds
        A.add(f"servo_{side}", "servo", part_number="23x8x27 mm 8 g digital servo",
              specs={"torque_kgcm": 1.6, "mass_g": 8, "dims_mm": [23, 8, 27.3],
                     "cg_mm": cg(hb.mean(0)[0] - 25, hb[1, 1] + 6, hb.mean(0)[2])},
              connects=[f"wing_mid_{side}", "fc"],
              notes="README: '2 Servo' (servo_dimension_perso 23x8x27.3 mm); drives the elevon printed in wing_mid")
    # CF spars (README: 5.5 mm and 6.5 mm tubes; 2 per half-wing + 1). Positions from OpenGeoDrone.scad
    # (spar_hole_perc 15 % / 37 % chord, 3rd spar at 75 %); wall thickness and exact lengths assumed.
    for side, sgn in (("L", 1), ("R", -1)):
        for sid, od, idd, ln, frac in (("front", 5.5, 3.5, 420, 0.15), ("rear", 5.5, 3.5, 330, 0.37)):
            A.add(f"spar_{sid}_{side}", "spar", role="main_spar" if sid == "front" else "rear_spar",
                  part_number=f"CF tube {od}/{idd} mm", material={"name": "CF tube"},
                  specs={"outer_d_mm": od, "inner_d_mm": idd, "length_mm": ln,
                         "mass_g": round(3.1416 / 4 * (od**2 - idd**2) * ln * 1.55e-3, 1),
                         "cg_mm": cg(-60 + frac * 210 + 0.3 * ln * 0.55, -6, sgn * (45 + ln / 2))},
                  notes="half-wing spar inserted into the center part; ID and length assumed")
    A.add("spar_center", "spar", role="rear_spar", part_number="CF tube 6.5/4.5 mm", material={"name": "CF tube"},
          specs={"outer_d_mm": 6.5, "inner_d_mm": 4.5, "length_mm": 2 * (38 + 215),
                 "mass_g": round(3.1416 / 4 * (6.5**2 - 4.5**2) * 2 * (38 + 215) * 1.55e-3, 1),
                 "cg_mm": cg(-60 + 0.75 * 210, -6, 0)},
          notes="3rd spar (75 % chord) through the center part and both root panels; ID assumed")
    A.add("wiring", "wire", part_number="silicone wiring + XT60", specs={"mass_g": 60, "cg_mm": cg(xc, 0, 0)},
          connects=["battery", "esc_4in1", "pusher_esc"])
    A.add("screws", "fastener", group="airframe_screws", qty=16, part_number="M3 screws + magnets",
          specs={"mass_g": 0.9, "cg_mm": cg(xc, 0, 0)})

    header = (f"OpenGeoDrone (github.com/{REPO} @ {COMMIT[:12]}, GPL-3.0), re-assembled by "
              "fixtures/fetch_third_party.py.\nCAD frame = OpenSCAD frame: mm, x aft, y up, z = left span. "
              "specs.cg_mm is in this CAD frame.\nParts without CAD come from the README hardware list; "
              "models/masses marked 'assumed' in notes.")
    mission = {"preset": "endurance", "payload_ids": [], "frame": frame}
    out = Path(out)
    A.write(out, mission, header)
    (out / "SOURCE.md").write_text(SOURCE_MD.format(commit=COMMIT, repo=REPO,
                                                    files="\n".join(f"- `3mf_print_part/X1C_{p}.3mf`" for p in PLATES)))
    tot = sum(f.stat().st_size for f in (out / "cad").iterdir())
    print(f"third_party: {len(A.parts)} parts, {len(A.meshes)} meshes, cad {tot / 1e6:.1f} MB -> {out}")


SOURCE_MD = """# third_party: OpenGeoDrone

- Source: https://github.com/{repo}
- Commit: `{commit}`
- License: GPL-3.0 (repository LICENSE). The meshes in `cad/` are derived from the repository's
  3MF print files and stay under GPL-3.0; keep this notice with them.
- Design: parametric vase-mode VTOL flying wing by ENAC UAV Lab (FlyHigh project, ANR-24-CE92-0015 /
  DFG GR 5407/2-1). Half-span 490 mm + 90 mm center body, root chord 210 mm, MH61 airfoil, 4 hover motors
  on two arms, 1 pusher, 2 elevons. README: mass without battery 1.066 kg, 4S 4 Ah battery.

## Files fetched (Bambu Studio X1C print plates, ~8.4 MB total)
{files}

The per-part STLs in `stl_print_part/` are ~190 MB, over our download budget, so the 3MF plates were used.
Each 3MF object is the part mesh (decimated by the slicer) re-centred on its bounding box; the plate
transforms (print orientation / bed position) were dropped, which keeps the original OpenSCAD axes.

## Re-assembly (approximate)
Translations were rebuilt in the OpenSCAD frame (mm, x aft, y up, z = left span):
- center part at the origin (symmetric about z = 0, width 90 = `center_width`);
- root panels stacked at |z| = 45..250 (`wing_root_mm` 215 - 10 mm hull), motor arms in the 38 mm gap
  (|z| = 250..288), mid+elevon panels at |z| = 288..535 (half-span 490 + 45);
- x/y of each panel matched to the neighbouring panel's airfoil section (bbox mid-point of the section at the
  joint; across the arm gap the root panel's LE/TE sweep is extrapolated);
- motor arm front/back halves meet at 40 % chord of the local section; fuselage front/bottom-back/upper are
  stacked as cut in `OpenGeoDrone.scad` (12 mm overlap) with the tail flush with the center part TE;
- 6 wing clamps placed at the joints; servo horns under the elevon roots.
Expect a few mm of error and some interpenetration (declared in `specs.fits`). The elevons are printed as
one body with the mid panels, so there is no separate control-surface part; as a tailless flying wing it
has no horizontal/vertical tail.

## Not in the CAD (from the README hardware list)
Hover motors (F40), 7x4 props, 4-in-1 ESC, pusher motor (< 1200 kV) + ESC + 9x6 prop, 4S 4 Ah battery,
Tawaki FC, RC receiver, GPS, telemetry modem, pitot, 2 servos, CF spars (5.5 / 6.5 mm), wiring, screws.
Their models, masses and positions are estimates (see `notes` in bom.yaml); `specs.cg_mm` is in the CAD frame.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None, help="download cache dir (default: a temp dir)")
    ap.add_argument("--out", default=str(HERE / "third_party"))
    a = ap.parse_args(argv)
    if a.cache:
        build(Path(a.out), Path(a.cache))
    else:
        with tempfile.TemporaryDirectory() as d:
            build(Path(a.out), Path(d))


if __name__ == "__main__":
    main()
