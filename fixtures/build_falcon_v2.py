"""Build fixtures/falcon_v2 (Titan Falcon V2 replica) from fixtures/falcon_v2/params.yaml.

    .venv/bin/python fixtures/build_falcon_v2.py [--params PATH] [--out DIR]

Geometry is built in metres (x aft, y right, z up, nose at x = 0) and written as mm STL.
Output is deterministic. See fixtures/README.md for the planform/CG assumptions.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import meshkit as mk  # noqa: E402
from dronebench.components import lookup  # noqa: E402

MM = 1e-3


def r1(v):
    return round(float(v), 1)


def mm3(p):
    return [r1(c / MM) for c in p]


class Planform:
    """Half-wing planform: constant chord to break_y, then linear taper; LE sweep outboard of the break."""

    def __init__(self, w, z_root):
        self.b2 = w["span_mm"] / 2 * MM
        self.cr, self.ct = w["root_chord_mm"] * MM, w["tip_chord_mm"] * MM
        self.yb = w["break_y_mm"] * MM
        self.sweep = math.radians(w["sweep_le_deg"])
        self.dih = math.radians(w["dihedral_deg"])
        self.x0 = w["x_le_root_mm"] * MM
        self.af0, self.af1 = w["root_airfoil"], w["tip_airfoil"]
        self.z0 = z_root

    def chord(self, y):
        y = abs(y)
        return self.cr if y <= self.yb else self.cr + (self.ct - self.cr) * (y - self.yb) / (self.b2 - self.yb)

    def x_le(self, y):
        return self.x0 + math.tan(self.sweep) * max(0.0, abs(y) - self.yb)

    def z(self, y):
        return self.z0 + abs(y) * math.tan(self.dih)

    def airfoil(self, y):
        f = max(0.0, (abs(y) - self.yb) / (self.b2 - self.yb))
        return mk.blend_airfoil(self.af0, self.af1, f)

    def station(self, y):
        return (y, self.chord(y), self.x_le(y), self.z(y), self.airfoil(y))

    def stations(self, ya, yb):
        """Stations from |y| = ya to yb (positive side), including the planform break if inside."""
        ys = [ya] + ([self.yb] if ya < self.yb < yb else []) + [yb]
        return [self.station(y) for y in ys]

    def area_mac(self, n=4000):
        ys = np.linspace(0, self.b2, n)
        c = np.array([self.chord(y) for y in ys])
        xle = np.array([self.x_le(y) for y in ys])
        half = np.trapezoid(c, ys)
        mac = np.trapezoid(c * c, ys) / half
        y_mac = np.trapezoid(c * ys, ys) / half
        x_le_mac = np.trapezoid(c * xle, ys) / half
        return 2 * half, mac, y_mac, x_le_mac


def mirror(stations):
    return [(-y, c, x, z, af) for (y, c, x, z, af) in stations][::-1]


def build(params: dict, out: Path, verbose=True):
    t0 = time.time()
    P = params
    W, H, V, F = P["wing"], P["htail"], P["vtail"], P["fuselage"]
    A = mk.Assembly()

    # ---------------- fuselage hull (outer profile). Wing saddle x 380..660 has a constant section.
    fw, fh = F["max_width_mm"] * MM, F["max_height_mm"] * MM
    zc_main = -0.045
    x_le = W["x_le_root_mm"] * MM
    cr = W["root_chord_mm"] * MM
    plan0 = Planform(W, 0.0)
    S, mac, y_mac, x_le_mac = plan0.area_mac()
    x_ac_w = x_le_mac + 0.25 * mac
    # horizontal tail geometry
    Sh, ARh, lam_h = H["area_m2"], H["aspect_ratio"], H["taper"]
    bh = math.sqrt(Sh * ARh)
    crh = 2 * Sh / (bh * (1 + lam_h))
    cth = crh * lam_h
    mach = 2 / 3 * crh * (1 + lam_h + lam_h**2) / (1 + lam_h)
    x_ac_h = x_ac_w + H["arm_mm"] * MM
    x_le_h = x_ac_h - 0.25 * mach        # unswept LE
    tail_end = x_le_h + crh + 0.010
    tail_h = 0.044
    stations = [
        (0.000, 0.040, 0.040, zc_main + 0.005),
        (0.080, fw * 0.83, fh * 0.85, zc_main),
        (0.200, fw, fh, zc_main),
        (x_le + cr + 0.020, fw, fh, zc_main),
        (x_le + cr + 0.160, fw * 0.66, fh * 0.70, zc_main + 0.017),
        (x_le_h - 0.060, tail_h, tail_h, -0.004),
        (tail_end, tail_h, tail_h, -0.004),
    ]
    hull = mk.Hull(stations, wall=F["wall_mm"] * MM)
    fuse_top = hull.top(x_le)

    # wing chord-line z at root: lowest point of the root airfoil touches the fuselage top (+0.3 mm)
    xs = mk.chord_stations(0, 1, 200)
    _, lo = mk.naca4_surfaces(W["root_airfoil"], xs)
    z_root = fuse_top - lo[:, 1].min() * cr + 0.0003
    plan = Planform(W, z_root)

    xb = [x * MM for x in F["sections_x_mm"]] + [tail_end]
    fuse_mat = {"name": "LW-PLA", "infill": 0.0, "walls": 3, "orientation": "flat"}
    for i in range(len(xb) - 1):
        A.add(f"fuse{i + 1}", "fuselage", hull.section(xb[i], xb[i + 1]), role="fuselage",
              part_number=f"Falcon V2 Fuse{i + 1}", material=fuse_mat,
              notes="printed LW-PLA fuselage section (hollow shell)")

    # ---------------- wing panels, flaps, ailerons
    br = [b * MM for b in W["panel_breaks_mm"]] + [plan.b2]
    cf_flap, cf_ail = W["flap_chord_frac"], W["aileron_chord_frac"]
    wmat = dict(W["material"])
    for side, sgn in (("L", -1), ("R", 1)):
        for i in range(3):
            st = plan.stations(br[i], br[i + 1])
            if sgn < 0:
                st = mirror(st)
            pid = f"wing_{side}{i + 1}"
            f1 = {0: 1.0, 1: 1 - cf_flap, 2: 1 - cf_ail}[i]
            A.add(pid, "wing", mk.loft_surface(st, 0.0, f1), role="wing",
                  part_number=f"Falcon V2 Wing{i + 1} {side}", material=wmat)
            if i == 1:
                A.add(f"flap_{side}", "surface", mk.loft_surface(st, f1, 1.0), role="flap", parent=pid,
                      part_number=f"Falcon V2 Flap {side}",
                      material={"name": "LW-PLA", "infill": 0.0, "walls": 1, "orientation": "vase"})
            if i == 2:
                A.add(f"aileron_{side}", "surface", mk.loft_surface(st, f1, 1.0), role="aileron", parent=pid,
                      part_number=f"Falcon V2 Aileron {side}",
                      material={"name": "LW-PLA", "infill": 0.0, "walls": 1, "orientation": "vase"})

    def hinge_pts(y0, y1, cf):
        pts = []
        for y in (y0, y1):
            c, xl, z, af = plan.chord(y), plan.x_le(y), plan.z(y), plan.airfoil(y)
            u, l = mk.naca4_surfaces(af, np.array([1 - cf]))
            pts.append(np.array([xl + (1 - cf) * c, y, z + 0.5 * (u[0, 1] + l[0, 1]) * c]))
        return pts

    def camber_z(y, frac):
        u, l = mk.naca4_surfaces(plan.airfoil(y), np.array([frac]))
        return plan.z(y) + 0.5 * (u[0, 1] + l[0, 1]) * plan.chord(y)

    # ---------------- spars (straight tubes; wing dihedral is small)
    sp = P["spars"]
    x_main = x_le + sp["main"]["chord_frac"] * cr
    z_main = camber_z(0, sp["main"]["chord_frac"])
    Lm = 2 * sp["main"]["half_length_mm"] * MM
    wing_ids = [f"wing_{s}{i}" for s in "LR" for i in (1, 2, 3)]
    A.add("main_spar", "spar", mk.make_tube(sp["main"]["od_mm"] * MM, sp["main"]["id_mm"] * MM, Lm,
                                            (x_main, 0, z_main), "y"),
          role="main_spar", part_number=f"CF tube {sp['main']['od_mm']}/{sp['main']['id_mm']} mm pultruded",
          material={"name": "CF tube"},
          specs={"outer_d_mm": sp["main"]["od_mm"], "inner_d_mm": sp["main"]["id_mm"], "length_mm": r1(Lm / MM),
                 "fits": wing_ids})
    x_rear = x_le + sp["rear"]["chord_frac"] * cr
    z_rear = camber_z(0, sp["rear"]["chord_frac"])
    Lr = 2 * sp["rear"]["half_length_mm"] * MM
    A.add("rear_spar", "spar", mk.make_tube(sp["rear"]["od_mm"] * MM, sp["rear"]["id_mm"] * MM, Lr,
                                            (x_rear, 0, z_rear), "y"),
          role="rear_spar", part_number=f"CF tube {sp['rear']['od_mm']}/{sp['rear']['id_mm']} mm pultruded",
          material={"name": "CF tube"},
          specs={"outer_d_mm": sp["rear"]["od_mm"], "inner_d_mm": sp["rear"]["id_mm"], "length_mm": r1(Lr / MM),
                 "fits": ["wing_L1", "wing_R1", "wing_L2", "wing_R2"]})

    # ---------------- nacelles, motors, props
    pr = P["propulsion"]
    mot, prop, esc = lookup(pr["motor"]), lookup(pr["prop"]), lookup(pr["esc"])
    ym = pr["motor_y_mm"] * MM
    nac_w = 0.042
    xs_n = mk.chord_stations(0, 1, 120)
    _, lo_ym = mk.naca4_surfaces(plan.airfoil(ym), xs_n)
    wing_low_ym = plan.z(ym) + lo_ym[:, 1].min() * plan.chord(ym)
    nac_zc = wing_low_ym - 0.0003 - nac_w / 2
    x_nac0 = x_le - 0.060
    motor_len, shaft_len, hub_len = 0.032, 0.012, 0.012
    for side, sgn in (("L", -1), ("R", 1)):
        y = sgn * ym
        A.add(f"nacelle_{side}", "fairing", mk.solid_body(
            [(x_nac0, nac_w * 0.95, nac_w * 0.95, nac_zc), (x_nac0 + 0.06, nac_w, nac_w, nac_zc),
             (x_nac0 + 0.20, nac_w * 0.8, nac_w * 0.8, nac_zc), (x_nac0 + 0.28, 0.012, 0.012, nac_zc + 0.01)],
            y=y), role="nacelle", part_number=f"Falcon V2 Nacelle {side}",
            material={"name": "PLA", "infill": 0.15, "walls": 2, "orientation": "flat"},
            notes="motor mount nacelle under the wing")
        mc = (x_nac0 - motor_len / 2, y, nac_zc)
        A.add(f"motor_{side}", "motor", mk.motor(mc, 0.035, motor_len, shaft_len), part_number=pr["motor"],
              specs=mot["specs"], connects=[f"prop_{side}"], material={"name": "Motor assembly", "sub": mot["sub"]})
        pc = (x_nac0 - motor_len - shaft_len - hub_len / 2 + 0.0002, y, nac_zc)
        A.add(f"prop_{side}", "prop", mk.prop(pc, prop["specs"]["diam_in"] * 0.0254, prop["specs"]["pitch_in"] * 0.0254,
                                             spin=-sgn),
              part_number=pr["prop"], specs=dict(prop["specs"], rotation="CCW" if sgn < 0 else "CW"),
              notes="tractor, in front of the wing LE")
        # ESC inside the nacelle (declared fit), just behind the motor
        A.add(f"esc_{side}", "esc", mk.box((0.055, 0.026, 0.012), (x_nac0 + 0.06, y, nac_zc)),
              part_number=pr["esc"], specs=dict(esc["specs"], fits=[f"nacelle_{side}"]),
              connects=[f"motor_{side}"])
        A.add(f"screws_motor_{side}", "fastener", group=f"motor_{side}_mount", qty=4, part_number="M3x8 steel screw",
              specs={"size": "M3x8", "mass_g": 0.8, "cg_mm": mm3((x_nac0 + 0.003, y, nac_zc))},
              connects=[f"motor_{side}", f"nacelle_{side}"],
              notes=f"fixes motor_{side} to nacelle_{side}")

    # ---------------- avionics tray + bay (inside Fuse2/Fuse3)
    tray_x0, tray_x1, tray_w, tray_t = 0.210, x_le + cr - 0.010, 0.064, 0.002
    tray_z = hull.floor(tray_x0, tray_x1, tray_w / 2) + 0.0002
    A.add("tray", "mount", mk.box((tray_x1 - tray_x0, tray_w, tray_t), ((tray_x0 + tray_x1) / 2, 0, tray_z + tray_t / 2)),
          part_number="Falcon V2 equipment tray",
          material={"name": "PLA", "infill": 0.2, "walls": 2, "orientation": "flat"})
    top = tray_z + tray_t

    def on_tray(id_, typ, size, x, y=0.0, **kw):
        A.add(id_, typ, mk.box(size, (x, y, top + size[2] / 2 + 0.0001)), **kw)

    fc = lookup("Matek F405-WTE")
    on_tray("fc", "fc", (0.044, 0.036, 0.012), tray_x1 - 0.048, part_number="Matek F405-WTE", specs=fc["specs"],
            connects=["esc_L", "esc_R"])
    rx = lookup("ELRS EP2")
    on_tray("rx", "rx", (0.018, 0.012, 0.004), tray_x1 - 0.012, 0.012, part_number="ExpressLRS EP2 receiver",
            specs=rx["specs"], connects=["fc"])
    gps = lookup("Matek M10Q-5883")
    on_tray("gps", "gps", (0.020, 0.020, 0.010), tray_x1 - 0.012, -0.018, part_number="Matek M10Q-5883",
            specs=gps["specs"], connects=["fc"])
    pit = lookup("Matek ASPD-4525")
    on_tray("pitot", "sensor", (0.024, 0.016, 0.010), 0.240, 0.020, part_number="Matek ASPD-4525",
            specs=pit["specs"], connects=["fc"])

    # battery (auto-placed after everything else exists)
    bat = lookup(P["battery"]["part_number"])
    bsz = [s * MM for s in P["battery"]["size_mm"]]

    # ---------------- payload camera in the nose (locked)
    cam = P["payload"]
    csz = [s * MM for s in cam["size_mm"]]
    cx0, cx1 = 0.040, 0.140
    cm_z = hull.floor(cx0, cx1, 0.020) + 0.0002
    A.add("camera_mount", "mount", mk.box((cx1 - cx0, 0.040, 0.003), ((cx0 + cx1) / 2, 0, cm_z + 0.0015)),
          part_number="Falcon V2 camera mount", material={"name": "PLA", "infill": 0.2, "walls": 2,
                                                            "orientation": "flat"})
    ctop = cm_z + 0.003
    A.add("camera", "camera", mk.box(csz, (0.105, 0, ctop + csz[2] / 2 + 0.0001)), part_number=cam["part_number"],
          specs={**(lookup(cam["part_number"]) or {}).get("specs", {}), "mass_g": cam["mass_g"]}, locked=True,
          connects=["fc"], notes="mission payload (mapping camera)")

    # ---------------- tail
    af_h = H["airfoil"]
    _, lo_h = mk.naca4_surfaces(af_h, mk.chord_stations(0, 1, 120))
    z_h = hull.top(x_le_h + 0.5 * crh) - lo_h[:, 1].min() * crh + 0.0003
    ce = H["elevator_chord_frac"]
    hst = [(-bh / 2, cth, x_le_h, z_h, af_h), (0.0, crh, x_le_h, z_h, af_h), (bh / 2, cth, x_le_h, z_h, af_h)]
    tmat = {"name": "LW-PLA", "infill": 0.0, "walls": 2, "orientation": "vase"}
    smat = {"name": "LW-PLA", "infill": 0.0, "walls": 1, "orientation": "vase"}
    A.add("hstab", "tail", mk.loft_surface(hst, 0, 1 - ce), role="htail", part_number="Falcon V2 Hstab",
          material=tmat)
    gap = 0.003
    ch_mid = crh + (cth - crh) * (gap / (bh / 2))
    A.add("elevator_L", "surface", mk.loft_surface([hst[0], (-gap, ch_mid, x_le_h, z_h, af_h)], 1 - ce, 1),
          role="elevator", parent="hstab", part_number="Falcon V2 Elevator L", material=smat)
    A.add("elevator_R", "surface", mk.loft_surface([(gap, ch_mid, x_le_h, z_h, af_h), hst[2]], 1 - ce, 1),
          role="elevator", parent="hstab", part_number="Falcon V2 Elevator R", material=smat)
    # vertical fin on top of the hstab root
    Sv, hv, lam_v = V["area_m2"], V["height_mm"] * MM, V["taper"]
    crv = 2 * Sv / (hv * (1 + lam_v))
    ctv = crv * lam_v
    up_h, _ = mk.naca4_surfaces(af_h, mk.chord_stations(0, 1, 120))
    z_v0 = z_h + up_h[:, 1].max() * crh + 0.0003
    x_le_v = x_le_h + crh - crv
    sweep_v = 0.35 * (crv - ctv) + 0.02
    vst = [(z_v0, crv, x_le_v, 0.0, V["airfoil"]), (z_v0 + hv, ctv, x_le_v + sweep_v, 0.0, V["airfoil"])]
    cr_f = V["rudder_chord_frac"]
    A.add("vstab", "tail", mk.loft_surface(vst, 0, 1 - cr_f, axis="z"), role="vtail",
          part_number="Falcon V2 Vstab", material=tmat)
    A.add("rudder", "surface", mk.loft_surface(vst, 1 - cr_f, 1, axis="z"), role="rudder", parent="vstab",
          part_number="Falcon V2 Rudder", material=smat)

    # ---------------- servos, horns, linkages, hinges
    servo_pn = P["servo"]
    sv = lookup(servo_pn)
    sdims = [d * MM for d in sv["specs"]["dims_mm"]]

    def servo_in_wing(sid, surf, parent, y, cf):
        c, xl, af = plan.chord(y), plan.x_le(y), plan.airfoil(y)
        xs_ = xl + (1 - cf) * c - 0.018
        zc_ = camber_z(y, (xs_ - xl) / c)
        m = mk.servo((xs_, y, zc_), sdims, orient="flat")
        A.add(sid, "servo", m, part_number=servo_pn, specs=dict(sv["specs"], fits=[parent]), connects=[surf, "fc"])
        hx = xl + (1 - cf) * c + 0.012
        zl = plan.z(y) + mk.naca4_surfaces(af, np.array([(hx - xl) / c]))[1][0, 1] * c
        horn = mk.box((0.010, 0.002, 0.012), (hx, y, zl - 0.006 + 0.001))
        A.add(f"horn_{surf}", "horn", horn, parent=surf, part_number="control horn (nylon)",
              material={"name": "Nylon PA66"}, specs={"fits": [surf]})
        link = mk.rod((xs_ + 0.010, y + 0.002, zl - 0.004), (hx, y + 0.002, zl - 0.009), 0.0012)
        A.add(f"link_{surf}", "linkage", link, part_number="1.2 mm steel pushrod + Z-bend",
              material={"name": "Steel"}, specs={"fits": [parent, surf, f"horn_{surf}", sid]}, connects=[sid, surf])

    for side, sgn in (("L", -1), ("R", 1)):
        yf = sgn * 0.5 * (br[1] + br[2])
        ya = sgn * (br[2] + 0.45 * (plan.b2 - br[2]))
        servo_in_wing(f"servo_flap_{side}", f"flap_{side}", f"wing_{side}2", yf, cf_flap)
        servo_in_wing(f"servo_aileron_{side}", f"aileron_{side}", f"wing_{side}3", ya, cf_ail)
        for surf, (ya_, yb_), cf, par in ((f"flap_{side}", (br[1], min(br[2], plan.yb)), cf_flap, f"wing_{side}2"),
                                          (f"aileron_{side}", (br[2], plan.b2 - 0.01), cf_ail, f"wing_{side}3")):
            p0, p1 = hinge_pts(sgn * (ya_ + 0.005), sgn * (yb_ - 0.005), cf)
            A.add(f"hinge_{surf}", "hinge", mk.rod(p0, p1, 0.0015), parent=surf,
                  part_number="1.5 mm steel hinge rod", material={"name": "Steel"}, specs={"fits": [surf, par]})
    # elevator servos in the hstab, rudder servo in the vstab root
    for side, sgn in (("L", -1), ("R", 1)):
        y = sgn * bh * 0.2
        c = crh + (cth - crh) * abs(y) / (bh / 2)
        xs_ = x_le_h + (1 - ce) * c - 0.018
        u, l = mk.naca4_surfaces(af_h, np.array([(xs_ - x_le_h) / c]))
        zc_ = z_h + 0.5 * (u[0, 1] + l[0, 1]) * c
        A.add(f"servo_elevator_{side}", "servo", mk.servo((xs_, y, zc_), sdims, orient="flat"), part_number=servo_pn,
              specs=dict(sv["specs"], fits=["hstab"]), connects=[f"elevator_{side}", "fc"])
        hx = x_le_h + (1 - ce) * c + 0.012
        zl = z_h + mk.naca4_surfaces(af_h, np.array([(hx - x_le_h) / c]))[1][0, 1] * c
        A.add(f"horn_elevator_{side}", "horn", mk.box((0.010, 0.002, 0.012), (hx, y, zl - 0.005)),
              parent=f"elevator_{side}", part_number="control horn (nylon)", material={"name": "Nylon PA66"},
              specs={"fits": [f"elevator_{side}"]})
        A.add(f"link_elevator_{side}", "linkage",
              mk.rod((xs_ + 0.010, y + 0.002, zl - 0.004), (hx, y + 0.002, zl - 0.008), 0.0012),
              part_number="1.2 mm steel pushrod + Z-bend", material={"name": "Steel"},
              specs={"fits": ["hstab", f"elevator_{side}", f"horn_elevator_{side}", f"servo_elevator_{side}"]},
              connects=[f"servo_elevator_{side}", f"elevator_{side}"])
        yy = (gap + 0.002, bh / 2 - 0.005)
        p0 = np.array([x_le_h + (1 - ce) * crh, sgn * yy[0], z_h + 0.5 * sum(
            x[0, 1] for x in mk.naca4_surfaces(af_h, np.array([1 - ce]))) * crh])
        p1 = np.array([x_le_h + (1 - ce) * cth, sgn * yy[1], p0[2] - 0.5 * sum(
            x[0, 1] for x in mk.naca4_surfaces(af_h, np.array([1 - ce]))) * (crh - cth)])
        A.add(f"hinge_elevator_{side}", "hinge", mk.rod(p0, p1, 0.0015), parent=f"elevator_{side}",
              part_number="1.5 mm steel hinge rod", material={"name": "Steel"},
              specs={"fits": [f"elevator_{side}", "hstab"]})
    zr = z_v0 + 0.25 * hv
    cv = crv + (ctv - crv) * 0.25
    xlv = x_le_v + sweep_v * 0.25
    xs_ = xlv + (1 - cr_f) * cv - 0.030
    A.add("servo_rudder", "servo", mk.servo((xs_, 0.0, zr), sdims, orient="x"), part_number=servo_pn,
          specs=dict(sv["specs"], fits=["vstab"]), connects=["rudder", "fc"])
    hx = xlv + (1 - cr_f) * cv + 0.012
    th = mk.naca4_surfaces(V["airfoil"], np.array([(hx - xlv) / cv]))[0][0, 1] * cv
    A.add("horn_rudder", "horn", mk.box((0.010, 0.012, 0.002), (hx, th + 0.005, zr)), parent="rudder",
          part_number="control horn (nylon)", material={"name": "Nylon PA66"}, specs={"fits": ["rudder"]})
    A.add("link_rudder", "linkage", mk.rod((xs_ + 0.010, th + 0.003, zr + 0.002), (hx, th + 0.008, zr + 0.002), 0.0012),
          part_number="1.2 mm steel pushrod + Z-bend", material={"name": "Steel"},
          specs={"fits": ["vstab", "rudder", "horn_rudder", "servo_rudder"]}, connects=["servo_rudder", "rudder"])
    q0 = np.array([x_le_v + (1 - cr_f) * crv, 0, z_v0 + 0.004])
    q1 = np.array([x_le_v + sweep_v + (1 - cr_f) * ctv, 0, z_v0 + hv - 0.004])
    A.add("hinge_rudder", "hinge", mk.rod(q0, q1, 0.0015), parent="rudder", part_number="1.5 mm steel hinge rod",
          material={"name": "Steel"}, specs={"fits": ["rudder", "vstab"]})
    fc_entry = A.get("fc")
    fc_entry["connects"] = fc_entry["connects"] + [p["id"] for p in A.parts if p["type"] == "servo"]

    # ---------------- fasteners (no CAD; cg given)
    A.add("wing_bolts", "fastener", group="wing_mount", qty=4, part_number="M5x30 nylon wing bolt",
          material={"name": "Nylon PA66"}, specs={"size": "M5x30", "mass_g": 1.2,
                                                  "cg_mm": mm3((x_le + 0.5 * cr, 0, fuse_top))},
          connects=["wing_L1", "wing_R1", "fuse2", "fuse3"],
          notes="wing panels to fuselage")
    A.add("servo_screws", "fastener", group="servo_mount", qty=14, part_number="M2x8 steel screw",
          specs={"size": "M2x8", "mass_g": 0.25, "cg_mm": mm3((x_le + 0.4, 0, z_root))},
          connects=["servo_flap_L", "servo_flap_R", "servo_aileron_L", "servo_aileron_R", "servo_elevator_L",
                    "servo_elevator_R", "servo_rudder"])
    A.add("tail_screws", "fastener", group="tail_mount", qty=4, part_number="M3x16 steel screw",
          specs={"size": "M3x16", "mass_g": 1.2, "cg_mm": mm3((x_le_h + 0.4 * crh, 0, z_h))},
          connects=["hstab", "vstab", "fuse5"])
    A.add("fuse_screws", "fastener", group="fuse_joints", qty=8, part_number="M3x10 steel screw",
          specs={"size": "M3x10", "mass_g": 0.9, "cg_mm": mm3((0.55, 0, zc_main))},
          connects=["fuse1", "fuse2", "fuse3", "fuse4", "fuse5"])
    A.add("wiring_power", "wire", part_number="12 AWG silicone wire + XT90/XT60 connectors",
          specs={"mass_g": 70, "gauge_awg": 12, "cg_mm": mm3((x_le + 0.05, 0, top + 0.01))},
          connects=["battery", "esc_L", "esc_R"])
    A.add("wiring_signal", "wire", part_number="servo extensions + signal leads",
          specs={"mass_g": 45, "gauge_awg": 26, "cg_mm": mm3((x_le + 0.25, 0, z_root))})

    # ---------------- seeded waste
    wst = P.get("waste", {})
    if wst.get("ballast_nose"):
        A.add("ballast_nose", "ballast", mk.box((0.030, 0.024, 0.0142), (0.058, 0, ctop + 0.0072)),
              part_number="80 g steel nose weight", material={"name": "Steel"}, specs={"mass_g": 80},
              notes="nose weight (legacy trim)")
    if wst.get("gps_2"):
        on_tray("gps_2", "gps", (0.020, 0.020, 0.010), tray_x1 - 0.083, 0.0, part_number="Matek M10Q-5883",
                specs=gps["specs"], connects=["fc"], notes="second GPS")
    if wst.get("spar_doubler"):
        A.add("spar_doubler", "spar", mk.make_tube(sp["main"]["id_mm"] * MM, (sp["main"]["id_mm"] - 2) * MM, 0.400,
                                                   (x_main, 0, z_main), "y"),
              role="main_spar", part_number=f"CF tube {sp['main']['id_mm']}/{sp['main']['id_mm'] - 2} mm doubler",
              material={"name": "CF tube"},
              specs={"outer_d_mm": sp["main"]["id_mm"], "inner_d_mm": sp["main"]["id_mm"] - 2, "length_mm": 400,
                     "fits": ["main_spar", "wing_L1", "wing_R1"]},
              notes="root sleeve inside main_spar")
    if wst.get("fairings"):
        for side, sgn in (("L", -1), ("R", 1)):
            y = sgn * (plan.b2 - 0.012)
            c, xl = plan.chord(y), plan.x_le(y)
            _, lo_t = mk.naca4_surfaces(plan.airfoil(y), mk.chord_stations(0, 1, 60))
            zt = plan.z(y) + lo_t[:, 1].min() * c - 0.0003 - 0.006
            A.add(f"fairing_{side}", "fairing", mk.solid_body(
                [(xl - 0.02, 0.004, 0.004, zt), (xl + 0.01, 0.014, 0.012, zt), (xl + c, 0.010, 0.010, zt),
                 (xl + c + 0.03, 0.003, 0.003, zt)], y=y),
                part_number=f"Falcon V2 wingtip fairing {side}", material={"name": "PLA", "infill": 0.15, "walls": 2,
                                                                          "orientation": "flat"},
                notes="cosmetic wingtip pod")
    if wst.get("fuse_armor"):
        ax0, ax1, aw, at = 0.230, 0.600, 0.080, 0.005
        az = hull.bottom(0.4) - 0.0003
        A.add("fuse_armor", "fairing", mk.box((ax1 - ax0, aw, at), ((ax0 + ax1) / 2, 0, az - at / 2)),
              part_number="PETG belly skid plate 5 mm",
              material={"name": "PETG", "infill": 0.5, "walls": 4, "orientation": "flat"},
              notes="belly protection plate")

    # ---------------- battery placement for the target CG
    x_cg_target = x_le_mac + P["target_cg_frac_mac"] * mac
    bx = P["battery"]["x_mm"]
    b_lo, b_hi = 0.252 + 0.002 + bsz[0] / 2, tray_x1 - 0.095 - bsz[0] / 2   # between pitot and gps_2/fc
    if bx == "auto":
        m0, cg0, _ = A.mass_cg()
        mb = bat["mass_g"]
        bx = (x_cg_target * (m0 + mb) - cg0[0] * m0) / mb
        bx_c = min(max(bx, b_lo), b_hi)
        if verbose and abs(bx_c - bx) > 1e-6:
            print(f"warning: battery x {bx * 1000:.0f} mm clamped to tray ({bx_c * 1000:.0f} mm)")
        bx = round(bx_c, 3)
    else:
        bx = bx * MM
    A.add("battery", "battery", mk.box(bsz, (bx, 0, top + bsz[2] / 2 + 0.0001)), part_number=P["battery"]["part_number"],
          specs=bat["specs"], connects=["esc_L", "esc_R", "fc"])

    # ---------------- write
    total, cg, rows = A.mass_cg()
    sm_note = (f"S={S:.4f} m2  MAC={mac * 1000:.1f} mm  x_le_mac={x_le_mac * 1000:.1f} mm  x_ac={x_ac_w * 1000:.1f} mm  "
               f"CG x={cg[0] * 1000:.1f} mm ({(cg[0] - x_le_mac) / mac:.3f} MAC)")
    header = f"""Titan Falcon V2 replica, generated by fixtures/build_falcon_v2.py from params.yaml. Do not hand-edit;
edit params.yaml and rebuild. Frame: mm, x aft (nose at 0), y right, z up.
{sm_note}
AUW estimate (dronebench.materials.part_mass_g): {total:.0f} g
specs.fits lists parts this part is allowed to sit inside (declared fits for the clearance check)."""
    mission = {"preset": "endurance", "payload_ids": ["camera"], "cruise_kmh": 55,
               "frame": {"units": "mm", "forward": "-x", "up": "+z"}}
    A.write(out, mission, header)
    info = {"S": S, "mac": mac, "x_le_mac": x_le_mac, "x_ac": x_ac_w, "mass_g": total, "cg": cg,
            "battery_x": bx, "Sh": Sh, "Sv": Sv, "x_ac_h": x_ac_h, "rows": rows, "time_s": time.time() - t0}
    if verbose:
        print(sm_note)
        print(f"AUW {total:.0f} g, {len(A.parts)} parts, battery x {bx * 1000:.0f} mm, built in {info['time_s']:.1f} s")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(HERE / "falcon_v2" / "params.yaml"))
    ap.add_argument("--out", default=str(HERE / "falcon_v2"))
    a = ap.parse_args(argv)
    params = yaml.safe_load(open(a.params))
    build(params, Path(a.out))


if __name__ == "__main__":
    main()
