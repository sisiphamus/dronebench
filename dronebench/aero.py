"""Aerodynamics from the measured geometry, using AeroSandbox instead of hand-fitted formulas.

    ap  = build_airplane(geom, cg_m)          # asb.Airplane: wing, tails, fuselage, control surfaces
    tab = AeroTable.build(ap, geom, v_ref)    # coefficient tables the flight simulator looks up

Which tool is used for what (each choice was tested against the alternatives on the Falcon V2 geometry):

  * Lift, drag and pitching moment against angle of attack, including stall: AeroSandbox AeroBuildup
    (NeuralFoil section polars + body build-up). Evaluated at three airspeeds so Reynolds-number effects
    on drag are real, not a fixed factor.
  * Stability and damping derivatives (CL_alpha, Cm_alpha, Cm_q, Cn_beta, Cl_p, Cn_r, ...): AeroSandbox
    vortex lattice. AeroBuildup is NOT used for these: it ignores the wing's downwash on the tail and put the
    neutral point at 31% MAC where the vortex lattice (and a hand calculation) give 22%.
  * Control effectiveness (aileron, elevator, rudder): AeroBuildup with the surfaces deflected. (The vortex
    lattice in this AeroSandbox version ignores deflection; tested, returns zero.)
  * Fuselage destabilising moment: Munk slender-body term, added to Cm_alpha because the vortex lattice
    has no fuselage.

Frame: x aft from the nose, y right, z up (design frame, also AeroSandbox's geometry frame). The simulator
converts to body axes (x forward, y right, z down) at its own boundary.

Not modelled, so nobody trusts it blindly: propeller slipstream over the wing, gaps and seams (a single
`cd0_factor` stands in for them and is calibrated, see calibrate.py), aeroelastic bending, ground effect.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import aerosandbox as asb

DEG = math.pi / 180.0
ALPHAS_DEG = np.array([-8, -5, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 25], float)
SPEEDS = np.array([8.0, 15.0, 28.0])          # m/s, Reynolds sweep for drag


_POOL = None


def _pool():
    """Forked worker pool (shares the already-imported AeroSandbox, so workers start instantly)."""
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        import os
        from concurrent.futures import ProcessPoolExecutor
        n = max(1, min(8, (os.cpu_count() or 2) - 1))
        _POOL = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork")) if n > 1 else False
    return _POOL


def _bu_task(args):
    ap, V, a, defl = args
    ap2 = ap.with_control_deflections(defl) if defl else ap
    r = asb.AeroBuildup(airplane=ap2, op_point=_op(V, a)).run()
    return {k: _f(r[k]) for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")}


def _map(tasks):
    pool = _pool()
    if pool:
        try:
            return list(pool.map(_bu_task, tasks, timeout=300))
        except Exception:      # a broken or stuck pool must never break an analysis
            global _POOL
            _POOL = None
            pass
    return [_bu_task(t) for t in tasks]


def _f(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _airfoil(name: Optional[str], default: str = "naca0012") -> asb.Airfoil:
    n = (name or default).lower().replace(" ", "")
    if not n.startswith("naca"):
        n = "naca" + n
    try:
        return asb.Airfoil(n)
    except Exception:
        return asb.Airfoil(default)


def _resample(stations, ys):
    st = sorted(stations, key=lambda s: s[0])
    y = np.array([s[0] for s in st]); c = np.array([s[1] for s in st])
    xl = np.array([s[2] for s in st]); z = np.array([s[3] for s in st])
    ys = np.clip(np.asarray(ys, float), y.min(), y.max())
    return ys, np.interp(ys, y, c), np.interp(ys, y, xl), np.interp(ys, y, z)


def _half_ys(half: float, breaks: list[float], n: int = 6) -> list[float]:
    pts = sorted(set([0.0, half] + [b for b in breaks if 0 < b < half] + list(np.linspace(0, half, n))))
    out = [pts[0]]
    for p in pts[1:]:
        if p - out[-1] > 0.03 * half:
            out.append(p)
    if out[-1] != half:
        out.append(half)
    return out


def _wing(surface, name: str, cs: Optional[dict] = None) -> asb.Wing:
    """Horizontal surface (both sides) from measured stations. cs = {"name": "aileron", "span": s, "hinge": h,
    "split": True} puts a control surface on the outer `span` of each side (split = antisymmetric _L/_R names)."""
    af = _airfoil(surface.airfoil)
    half = surface.span_m / 2
    brk = [half - cs["span"]] if cs else []
    hy = _half_ys(half, brk)
    ys = [-y for y in hy[::-1]] + hy[1:]
    ys, ch, xl, zz = _resample(surface.stations, ys)
    xsecs = []
    for i, (y, c, x, z) in enumerate(zip(ys, ch, xl, zz)):
        surfs = []
        if cs and i + 1 < len(ys):
            a = half - cs["span"]
            if cs["split"]:
                if y >= a - 1e-6 and y < half - 1e-6:
                    surfs = [asb.ControlSurface(name=cs["name"] + "_R", symmetric=False, hinge_point=cs["hinge"])]
                if y <= -a + 1e-6 and y > -half - 1e-6 and ys[i + 1] <= -a + 1e-6 + 1e-9:
                    surfs = [asb.ControlSurface(name=cs["name"] + "_L", symmetric=False, hinge_point=cs["hinge"])]
            elif abs(y) >= a - 1e-6 and abs(ys[i + 1]) >= a - 1e-6 and abs(y) < half:
                surfs = [asb.ControlSurface(name=cs["name"], symmetric=True, hinge_point=cs["hinge"])]
        xsecs.append(asb.WingXSec(xyz_le=[float(x), float(y), float(z)], chord=max(float(c), 0.012), airfoil=af,
                                  control_surfaces=surfs))
    return asb.Wing(name=name, symmetric=False, xsecs=xsecs)


def _fin(surface, cs: Optional[dict]) -> asb.Wing:
    af = _airfoil(surface.airfoil)
    st = sorted(surface.stations, key=lambda s: s[0])
    h0, h1 = st[0][0], st[-1][0]
    hs = np.linspace(h0, h1, 7)
    _, ch, xl, zz = _resample(surface.stations, hs)
    xsecs = []
    for i in range(len(hs)):
        surfs = []
        if cs and i + 1 < len(hs) and hs[i] >= h0 + 0.02 * (h1 - h0) * 0:
            surfs = [asb.ControlSurface(name=cs["name"], symmetric=True, hinge_point=cs["hinge"])]
        xsecs.append(asb.WingXSec(xyz_le=[float(xl[i]), 0.0, float(zz[i])], chord=max(float(ch[i]), 0.012), airfoil=af,
                                  control_surfaces=surfs))
    return asb.Wing(name="vtail", symmetric=False, xsecs=xsecs)


def _fuselage(geom, z_axis: float) -> asb.Fuselage:
    L = max(geom.fuselage_length_m, 0.3)
    D = max(geom.fuselage_max_diam_m, 0.04)
    prof = [(0.0, 0.0), (0.03, 0.55), (0.12, 0.95), (0.25, 1.0), (0.60, 1.0), (0.85, 0.45), (1.0, 0.12)]
    xsecs = [asb.FuselageXSec(xyz_c=[f * L, 0.0, z_axis], radius=max(r * D / 2, 0.002)) for f, r in prof]
    return asb.Fuselage(name="fuselage", xsecs=xsecs)


def build_airplane(geom, cg_m) -> asb.Airplane:
    """AeroSandbox airplane from an AeroGeometry. The moment reference is `cg_m` (design frame)."""
    w, h, v = geom.wing, geom.htail, geom.vtail
    if w is None or h is None or v is None:
        raise ValueError("flight analysis needs a wing, a horizontal tail and a vertical tail")
    ail = [c for c in geom.control_surfaces if c.role == "aileron"]
    ele = [c for c in geom.control_surfaces if c.role in ("elevator",)]
    rud = [c for c in geom.control_surfaces if c.role in ("rudder",)]

    def chord_frac(cs_list, surf, y0=None, y1=None):
        """Moving-surface chord as a fraction of the local surface chord (hinge line = 1 - fraction)."""
        if not cs_list or surf is None:
            return 0.25
        c_mean = float(np.mean([c.chord_m for c in cs_list]))
        st = [(abs(q[0]), q[1]) for q in surf.stations if y0 is None or (y0 <= abs(q[0]) <= y1)] or [(0.0, surf.mac_m)]
        local = float(np.mean([c for _, c in st]))
        return float(np.clip(c_mean / max(local, 1e-3), 0.10, 0.50))

    half = w.span_m / 2
    a_span = max((c.span_m for c in ail), default=0.0)
    wing = _wing(w, "wing", {"name": "aileron", "span": a_span, "hinge": 1.0 - chord_frac(ail, w, half - a_span, half),
                             "split": True} if ail else None)
    htail = _wing(h, "htail", {"name": "elevator", "span": h.span_m / 2, "hinge": 1.0 - chord_frac(ele, h), "split": False} if ele else None)
    fin = _fin(v, {"name": "rudder", "hinge": 1.0 - chord_frac(rud, v)} if rud else None)
    return asb.Airplane(name="design", wings=[wing, htail, fin], fuselages=[_fuselage(geom, float(cg_m[2]))],
                        xyz_ref=[float(cg_m[0]), float(cg_m[1]), float(cg_m[2])],
                        s_ref=w.area_m2, c_ref=w.mac_m, b_ref=w.span_m)


def geometry_key(geom) -> str:
    """Hash of everything that changes the aerodynamics. Mass-only edits keep the same key (the table is reused)."""
    def surf(s):
        return None if s is None else [round(s.span_m, 4), round(s.area_m2, 5), s.airfoil,
                                       [[round(float(x), 4) for x in st] for st in s.stations[::4]]]
    blob = json.dumps({"w": surf(geom.wing), "h": surf(geom.htail), "v": surf(geom.vtail),
                       "fl": round(geom.fuselage_length_m, 3), "fd": round(geom.fuselage_max_diam_m, 3),
                       "cs": sorted([(c.role, round(c.span_m, 3), round(c.chord_m, 3)) for c in geom.control_surfaces]),
                       "v": 3},
                      sort_keys=True, default=float)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _op(v, alpha=0.0, beta=0.0, p=0.0, q=0.0, r=0.0, alt=0.0):
    return asb.OperatingPoint(atmosphere=asb.Atmosphere(altitude=alt), velocity=v, alpha=alpha, beta=beta, p=p, q=q, r=r)


def extra_drag_cda(geom, cf: float = 0.0055, form: float = 1.4) -> float:
    """Parasite drag the stations do not see (motor pods, GPS, pitot), CdA in m^2: wetted area x Cf x FF / 2."""
    s = geom.wetted_area_m2.get("nacelles", 0.0) + 0.35 * geom.wetted_area_m2.get("other", 0.0)
    return cf * form * s * 0.5


@dataclass
class AeroTable:
    """Everything the simulator needs about one outer shape. Moments are about `ref` (design frame, m)."""
    key: str
    s_ref: float
    b_ref: float
    c_ref: float
    ref: np.ndarray                      # moment reference point (x aft, y, z up), m
    alphas: np.ndarray                   # rad
    speeds: np.ndarray                   # m/s
    CL: np.ndarray                       # (speeds, alphas)
    CD: np.ndarray
    Cm: np.ndarray
    d: dict = field(default_factory=dict)        # derivative name -> value (per rad or per rad/s nondimensional)
    alpha_stall: float = 0.28
    cl_max: float = 1.4
    cda_extra: float = 0.0
    cd0_factor: float = 1.0
    notes: list = field(default_factory=list)

    # ---- lookups (alpha in rad)
    def _tab(self, tab, alpha, V):
        # linear in alpha, log-linear in speed
        i = np.clip(np.searchsorted(self.alphas, alpha) - 1, 0, len(self.alphas) - 2)
        t = (alpha - self.alphas[i]) / (self.alphas[i + 1] - self.alphas[i])
        row = tab[:, i] * (1 - t) + tab[:, i + 1] * t
        lv = np.log(np.clip(V, self.speeds[0], self.speeds[-1]))
        return float(np.interp(lv, np.log(self.speeds), row))

    def coeffs(self, alpha, beta, p, q, r, V, da=0.0, de=0.0, dr=0.0):
        """(CL, CD, CY, Cl, Cm, Cn) in stability axes. Rates p,q,r in rad/s, deflections in rad (positive = roll right,
        pitch up, yaw right command)."""
        d = self.d
        vv = max(V, 1.0)
        ph, qh, rh = p * self.b_ref / (2 * vv), q * self.c_ref / (2 * vv), r * self.b_ref / (2 * vv)
        cl = self._tab(self.CL, alpha, V) + d["CLq"] * qh + d["CLde"] * de
        cd = self._tab(self.CD, alpha, V) * self.cd0_factor + self.cda_extra / self.s_ref * self.cd0_factor
        # lateral-directional derivatives fade past stall (the flow no longer follows the surfaces)
        f = float(np.clip(1.0 - (alpha - self.alpha_stall) / 0.18, 0.25, 1.0)) if alpha > self.alpha_stall else 1.0
        cy = f * (d["CYb"] * beta + d["CYp"] * ph + d["CYr"] * rh + d["CYdr"] * dr)
        cll = f * (d["Clb"] * beta + d["Clp"] * ph + d["Clr"] * rh + d["Clda"] * da + d["Cldr"] * dr)
        cm = self._tab(self.Cm, alpha, V) + d["Cmq"] * qh + d["Cmde"] * de
        cn = f * (d["Cnb"] * beta + d["Cnp"] * ph + d["Cnr"] * rh + d["Cnda"] * da + d["Cndr"] * dr)
        cd += d.get("CDde", 0.0) * abs(de) + d.get("CDda", 0.0) * abs(da)
        return cl, cd, cy, cll, cm, cn

    # ---- derived numbers
    def static_margin(self, cg_x: float) -> float:
        """Neutral point margin in fractions of MAC, for a CG at design-frame x = cg_x."""
        return float(self.d["x_np"] - cg_x) / self.c_ref

    @staticmethod
    def build(ap: asb.Airplane, geom, key: str = "", v_ref: float = 15.0, cd0_factor: float = 1.0,
              resolution=(2, 4)) -> "AeroTable":
        notes = []
        # 1) nonlinear polars (AeroBuildup) at three speeds, computed in parallel
        grid = [(ap, float(V), float(a), None) for V in SPEEDS for a in ALPHAS_DEG]
        defl = 8.0
        ctl = [(ap, v_ref, 3.0, None), (ap, v_ref, 3.0, {"aileron_R": defl, "aileron_L": -defl}),
               (ap, v_ref, 3.0, {"elevator": defl}), (ap, v_ref, 3.0, {"rudder": defl})]
        out = _map(grid + ctl)
        CL = np.zeros((len(SPEEDS), len(ALPHAS_DEG))); CD = np.zeros_like(CL); Cm = np.zeros_like(CL)
        for idx, r in enumerate(out[:len(grid)]):
            i, j = divmod(idx, len(ALPHAS_DEG))
            CL[i, j], CD[i, j], Cm[i, j] = r["CL"], r["CD"], r["Cm"]
        base, r_ail, r_ele, r_rud = out[len(grid):]
        # 2) linear stability and damping derivatives (vortex lattice), at a lifting trim point
        vlm = asb.VortexLatticeMethod(airplane=ap, op_point=_op(v_ref, 3.0), spanwise_resolution=resolution[0],
                                      chordwise_resolution=resolution[1])
        sd = vlm.run_with_stability_derivatives()
        g = lambda k: _f(sd[k])
        d = {k: g(k) for k in ("CLa", "Cma", "CYb", "Clb", "Cnb", "CYp", "Clp", "Cnp", "CYr", "Clr", "Cnr", "CLq", "Cmq")}
        # fuselage: Munk slender-body destabilising moment (dCm/dalpha = 2 (k2-k1) Vol / (S c)), k2-k1 ~ 0.9
        L = max(geom.fuselage_length_m, 0.3); D = max(geom.fuselage_max_diam_m, 0.04)
        vol = 0.6 * math.pi / 4 * D * D * L
        cma_fus = 2 * 0.9 * vol / (ap.s_ref * ap.c_ref)
        d["Cma_fus"] = cma_fus
        lt = max(float(geom.htail.x_ac_m) - float(ap.xyz_ref[0]), 0.1)
        d["CLq_vlm"] = d["CLq"]
        d["CLq"] = -d["Cmq"] / (lt / ap.c_ref)       # same tail force, different moment arm
        d["Cma_vlm"] = d["Cma"]
        d["Cma"] = d["Cma"] + cma_fus
        d["x_np"] = float(ap.xyz_ref[0]) - d["Cma"] / d["CLa"] * ap.c_ref
        # AeroBuildup has no wing-on-tail downwash, and its pitching moment is a difference of large terms that is
        # kinked near trim (the section polars switch regimes). Below the stall the table therefore follows the vortex
        # lattice slope exactly (a straight line through the build-up value at 3 degrees); the build-up's own departure
        # from linear (the stall break) is blended in over the last 6 degrees before CL max. The same line is what the
        # static-margin number reports, so the gate and the flown aircraft agree.
        a_rad = ALPHAS_DEG * DEG
        j0 = int(np.argmin(np.abs(ALPHAS_DEG - 3.0)))
        k_stall = int(np.argmax(CL[1]))
        a_stall = ALPHAS_DEG[k_stall]
        for i in range(len(SPEEDS)):
            sel = (ALPHAS_DEG >= 0) & (ALPHAS_DEG <= 6)
            slope_bu = np.polyfit(a_rad[sel], Cm[i, sel], 1)[0]
            lin_bu = Cm[i, j0] + slope_bu * (a_rad - a_rad[j0])
            lin_vlm = Cm[i, j0] + d["Cma"] * (a_rad - a_rad[j0])
            w = np.clip((ALPHAS_DEG - (a_stall - 6.0)) / 6.0, 0.0, 1.0)
            w = np.where(ALPHAS_DEG < a_stall - 6.0, 0.0, w)
            Cm[i] = lin_vlm + w * (Cm[i] - lin_bu)
        # neutral point of the table the simulator flies: pitching-moment slope over lift slope in the flying range
        sel = (ALPHAS_DEG >= 0) & (ALPHAS_DEG <= 6)
        cla_tab = float(np.polyfit(a_rad[sel], CL[1, sel], 1)[0])
        d["CLa_table"] = cla_tab
        d["x_np"] = float(ap.xyz_ref[0]) - d["Cma"] / cla_tab * ap.c_ref
        # 3) control effectiveness (AeroBuildup, deflected surfaces; computed above)
        names = {w.name for wing in ap.wings for x in wing.xsecs for w in x.control_surfaces}

        def delta(r):
            return {k: (r[k] - base[k]) / (defl * DEG) for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")}
        da_ = delta(r_ail) if "aileron_R" in names else {}
        de_ = delta(r_ele) if "elevator" in names else {}
        dr_ = delta(r_rud) if "rudder" in names else {}
        z = {k: 0.0 for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")}
        da_, de_, dr_ = (da_ or z), (de_ or z), (dr_ or z)
        # sign convention: positive command = roll right (right wing down), nose up, nose right.
        # asb body axes: x fwd, y right, z down, so a positive Cl rolls right wing down: aileron_R down + L up -> right wing UP,
        # hence the roll command flips the sign; elevator trailing-edge-down is nose DOWN (Cm < 0 per deflection): flip too.
        d["Clda"] = -da_["Cl"]; d["Cnda"] = -da_["Cn"]; d["CDda"] = abs(da_["CD"])
        d["Cmde"] = -de_["Cm"]; d["CLde"] = -de_["CL"]; d["CDde"] = abs(de_["CD"])
        k = 1.0 if dr_["Cn"] >= 0 else -1.0     # command convention: positive rudder = nose right = positive Cn
        d["CYdr"] = k * dr_["CY"]; d["Cldr"] = k * dr_["Cl"]; d["Cndr"] = k * dr_["Cn"]
        # 4) stall
        cl_row = CL[1]
        k = int(np.argmax(cl_row))
        tab = AeroTable(key=key, s_ref=ap.s_ref, b_ref=ap.b_ref, c_ref=ap.c_ref, ref=np.array(ap.xyz_ref, float),
                        alphas=a_rad, speeds=SPEEDS, CL=CL, CD=CD, Cm=Cm, d=d, alpha_stall=float(a_rad[k]),
                        cl_max=float(cl_row[k]), cda_extra=extra_drag_cda(geom), cd0_factor=cd0_factor, notes=notes)
        return tab


_CACHE: dict = {}


def _cache_dir():
    import os
    from pathlib import Path
    d = Path(os.environ.get("DRONEBENCH_CACHE", Path.home() / ".cache" / "dronebench")) / "aero"
    d.mkdir(parents=True, exist_ok=True)
    return d


def table_for(geom, cg_m, v_ref: float = 15.0) -> AeroTable:
    """AeroTable for this outer shape, cached in memory and on disk (keyed by the geometry hash).

    Tables hold forces and moments about the build CG; the simulator transfers moments to the real CG, so
    mass-only edits (battery, ballast, walls, removing a bracket) never trigger a rebuild."""
    import pickle
    key = geometry_key(geom)
    if key in _CACHE:
        return _CACHE[key]
    f = _cache_dir() / f"{key}.pkl"
    if f.exists():
        try:
            _CACHE[key] = pickle.loads(f.read_bytes())
            return _CACHE[key]
        except Exception:
            pass
    ap = build_airplane(geom, cg_m)
    tab = AeroTable.build(ap, geom, key=key, v_ref=v_ref)
    _CACHE[key] = tab
    try:
        f.write_bytes(pickle.dumps(tab))
    except Exception:
        pass
    return tab
