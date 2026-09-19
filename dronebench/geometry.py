"""Aerodynamic geometry and mass properties measured from the ingested meshes (frame: x aft, y right, z up).

Surface groups (by ``Part.role``); moving surfaces count in their parent's planform:
  wing  = wing + aileron + flap + elevon
  htail = htail + elevator
  vtail = vtail + rudder
  V-tail (vtail_ruddervator + ruddervator) is reported as its two projections: ``htail`` from slicing
  normal to y (horizontal projection) and ``vtail`` from slicing normal to z (side projection; area is
  the sum over both panels, MAC/AR are per panel). Its thickness/camber are measured on oblique cuts, so
  they are inflated by ~1/cos(dihedral).

Slicing: each group's meshes are cut by ``N_STATIONS`` planes at strip midpoints across the span (normal
to y; normal to z for a fin). Per station: chord = x extent of the section, x_le = min x, z = mid-height,
plus airfoil thickness and camber from binned upper/lower surfaces. Stations with no section get x_le/z
interpolated from their neighbours; their chord is interpolated too when the gap is a panel joint
(<= GAP_BRIDGE_M) or lies within the fuselage's y extent (so the area is the reference area carried
through the fuselage), else it is zero (a missing panel removes its area).
Integrals use the midpoint rule over the full span: S = sum(c) dy, MAC = sum(c^2) dy / S,
x_le_mac = sum(c x_le) dy / S, x_ac = x_le_mac + MAC/4, z_m = sum(c z) dy / S. Root/tip chords are
linear extrapolations of the two innermost/outermost stations to the centreline/tip. Sweep (LE) and
dihedral are least-squares fits of x_le and z against |y - centreline| over measured stations; a fin
reports dihedral 0 (not defined). The airfoil name is a NACA 4-digit guess from the innermost measured
station; ``thickness_ratio`` and ``camber`` are medians over measured stations.
SurfaceGeom.stations holds (span coordinate, chord, x_le, z) with the span coordinate = y (z for a fin).

Centreline: bbox centre y of the fuselage parts, else of all meshed parts.

Wetted areas (mesh surface area, m^2): wing, htail (incl. V-tail), vtail, fuselage (fuselage/hatch),
nacelles (role nacelle), other (meshed non-aero parts, except props, whose bbox is not inside the bbox of
an aero/fuselage/nacelle part, i.e. parts that stick out into the flow).

Surface measurement is cached on a hash of the aero/fuselage parts' ids, roles and mesh vertices, so the
removal test only re-measures when an aero part changes.

Also public: watertight mesh builders for tests and fixtures (``make_naca4_wing``, ``make_naca4_panel``,
``make_naca4_fin``, ``make_fuselage``, ``make_tube``), all in the DroneBench frame and metres.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np
import trimesh

from dronebench.model import AeroGeometry, ControlSurfaceGeom, Design, MassProps, Part, SurfaceGeom

N_STATIONS = 40
N_CS_STATIONS = 12
N_AIRFOIL_BINS = 30
GAP_BRIDGE_M = 0.01          # spanwise gaps up to this wide (panel joints) are bridged
WING_ROLES = ("wing", "aileron", "flap", "elevon")
HTAIL_ROLES = ("htail", "elevator")
VTAIL_ROLES = ("vtail", "rudder")
VEE_ROLES = ("vtail_ruddervator", "ruddervator")
CONTROL_PARENT = {"aileron": "wing", "flap": "wing", "elevon": "wing", "elevator": "htail",
                  "ruddervator": "htail", "rudder": "vtail"}
FUSELAGE_TYPES = ("fuselage", "hatch")
_CACHE_SIZE = 64
_cache: OrderedDict[str, AeroGeometry] = OrderedDict()


# --------------------------------------------------------------------------- mass

def mass_properties(design: Design) -> MassProps:
    """Total mass (kg) and mass-weighted CG over all parts (Part.mass_g x qty; parts without cg add mass only)."""
    by_part = {pid: (p.mass_g or 0.0) * p.qty for pid, p in design.parts.items()}
    total_g = sum(by_part.values())
    placed = [(by_part[pid], p.cg_m) for pid, p in design.parts.items() if p.cg_m is not None and by_part[pid] > 0]
    m_placed = sum(m for m, _ in placed)
    cg = sum(m * np.asarray(c, dtype=float) for m, c in placed) / m_placed if m_placed > 0 else np.zeros(3)
    return MassProps(total_kg=total_g / 1e3, cg_m=np.asarray(cg, dtype=float), by_part_g=by_part)


# --------------------------------------------------------------------------- slicing

def _sections(mesh: trimesh.Trimesh, axis: int, positions: np.ndarray) -> list[np.ndarray | None]:
    """Section points (k, 3) of ``mesh`` cut by planes normal to ``axis`` at ``positions``."""
    normal = np.zeros(3)
    normal[axis] = 1.0
    lines, to_3d, _ = trimesh.intersections.mesh_multiplane(mesh, np.zeros(3), normal, positions)
    out: list[np.ndarray | None] = []
    for seg, T in zip(lines, to_3d):
        if len(seg) < 2:
            out.append(None)
            continue
        pts2 = seg.reshape(-1, 2)
        out.append(trimesh.transform_points(np.column_stack([pts2, np.zeros(len(pts2))]), T))
    return out


def _airfoil(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    """(max thickness / c, max camber / c, x/c of max camber) of a section; u chordwise, v normal to chord."""
    i_le, i_te = int(np.argmin(u)), int(np.argmax(u))
    c = u[i_te] - u[i_le]
    xn = (u - u[i_le]) / c
    bins = np.minimum((xn * N_AIRFOIL_BINS).astype(int), N_AIRFOIL_BINS - 1)
    upper = np.full(N_AIRFOIL_BINS, -np.inf)
    lower = np.full(N_AIRFOIL_BINS, np.inf)
    np.maximum.at(upper, bins, v)
    np.minimum.at(lower, bins, v)
    ok = upper > lower
    if not ok.any():
        return 0.0, 0.0, 0.0
    centres = (np.arange(N_AIRFOIL_BINS) + 0.5) / N_AIRFOIL_BINS
    upper, lower, centres = upper[ok], lower[ok], centres[ok]
    camber = ((upper + lower) / 2 - (v[i_le] + (v[i_te] - v[i_le]) * centres)) / c
    j = int(np.argmax(np.abs(camber)))
    return float((upper - lower).max() / c), float(camber[j]), float(centres[j])


def naca_name(thickness: float, camber: float, camber_pos: float) -> str:
    """Nearest NACA 4-digit designation, e.g. (0.12, 0.04, 0.4) -> 'NACA4412'."""
    m = int(np.clip(round(camber * 100), 0, 9))
    p = int(np.clip(round(camber_pos * 10), 0, 9)) if m > 0 else 0
    tt = int(np.clip(round(thickness * 100), 1, 99))
    return f"NACA{m}{p}{tt:02d}"


def _profile(mesh: trimesh.Trimesh, axis: int, n: int) -> dict[str, np.ndarray] | None:
    """Per-station chord, x_le, height and airfoil data over the ``axis`` extent of ``mesh``."""
    lo, hi = mesh.bounds[:, axis]
    if hi - lo <= 0:
        return None
    d = (hi - lo) / n
    pos = lo + d * (np.arange(n) + 0.5)
    thick_axis = 2 if axis == 1 else 1
    rows = []
    for s, p in zip(_sections(mesh, axis, pos), pos):
        if s is None:
            continue
        u, v = s[:, 0], s[:, thick_axis]
        chord = u.max() - u.min()
        if chord <= 0:
            continue
        rows.append((p, chord, u.min(), (v.max() + v.min()) / 2, *_airfoil(u, v)))
    if not rows:
        return None
    r = np.array(rows)
    measured = np.isin(pos, r[:, 0])
    prof = {"pos": pos, "d": np.array(d), "measured": measured, "lo": np.array(lo), "hi": np.array(hi)}
    for k, name in enumerate(("chord", "x_le", "h")):
        prof[name] = np.interp(pos, r[:, 0], r[:, k + 1])
    for k, name in enumerate(("t", "m", "p")):
        prof[name] = r[:, k + 4]
    return prof


def _extrapolate(eta: np.ndarray, val: np.ndarray, target: float, d: float) -> float:
    """Linear extrapolation of ``val`` to ``eta = target`` from the two nearest stations at distinct eta.

    Mirror stations of a symmetric surface share eta, so the second station must differ by > d/2.
    """
    order = np.argsort(np.abs(eta - target), kind="stable")
    i = order[0]
    rest = [j for j in order[1:] if abs(eta[j] - eta[i]) > d / 2]
    if not rest:
        return float(val[i])
    j = rest[0]
    return float(val[i] + (val[j] - val[i]) * (target - eta[i]) / (eta[j] - eta[i]))


def _fit_angle(eta: np.ndarray, val: np.ndarray) -> float:
    if len(np.unique(eta)) < 2:
        return 0.0
    return float(np.degrees(np.arctan(np.polyfit(eta, val, 1)[0])))


def _bridged(pos: np.ndarray, parts: list[Part], axis: int, fuselage: tuple[float, float] | None) -> np.ndarray:
    """Stations that count as surface: inside a part's extent, in a panel joint gap of <= GAP_BRIDGE_M, or
    in a gap lying within the fuselage's extent (the wing carried through the fuselage)."""
    spans = sorted((float(p.bbox_m[0, axis]), float(p.bbox_m[1, axis])) for p in parts)
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1] + GAP_BRIDGE_M:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    keep = list(merged)
    if fuselage is not None:
        f0, f1 = fuselage[0] - GAP_BRIDGE_M, fuselage[1] + GAP_BRIDGE_M
        keep += [[a[1], b[0]] for a, b in zip(merged, merged[1:]) if f0 <= a[1] and b[0] <= f1]
    return np.any([(pos >= a) & (pos <= b) for a, b in keep], axis=0)


def _measure_surface(name: str, parts: list[Part], axis: int, centre: float, panels: int = 1,
                     fuselage: tuple[float, float] | None = None) -> SurfaceGeom | None:
    """Planform of one surface group; ``axis`` 1 = span along y (symmetric about ``centre``), 2 = fin along z.

    ``panels`` multiplies the area only (V-tail side projection); MAC and AR stay per panel. ``fuselage`` is
    the fuselage's extent along ``axis``: gaps inside it are bridged, other gaps (a missing panel) have
    zero chord.
    """
    mesh = trimesh.util.concatenate([p.mesh for p in parts])
    prof = _profile(mesh, axis, N_STATIONS)
    if prof is None:
        return None
    pos, d, chord, x_le, h, meas = prof["pos"], float(prof["d"]), prof["chord"], prof["x_le"], prof["h"], prof["measured"]
    chord = np.where(meas | _bridged(pos, parts, axis, fuselage), chord, 0.0)
    lo, hi = float(prof["lo"]), float(prof["hi"])
    if axis == 1:
        eta = np.abs(pos - centre)
        root_eta = 0.0 if lo < centre < hi else float(eta.min() - d / 2)
        z_col = h                                  # section mid-height
    else:
        eta = pos - lo
        root_eta = 0.0
        z_col = pos
    area = float(chord.sum() * d)
    mac = float((chord ** 2).sum() * d / area)
    x_le_mac = float((chord * x_le).sum() * d / area)
    span = hi - lo
    root = max(_extrapolate(eta, chord, root_eta, d), 0.0)
    tip = max(_extrapolate(eta, chord, float(eta.max() + d / 2), d), 0.0)
    inner = int(np.argmin(np.where(meas, eta, np.inf)))
    inner_m = int(np.sum(meas[:inner + 1])) - 1    # index of that station among measured rows
    thickness, camber, camber_pos = prof["t"], prof["m"], prof["p"]
    return SurfaceGeom(
        name=name,
        span_m=span,
        area_m2=area * panels,
        mac_m=mac,
        root_chord_m=root,
        tip_chord_m=tip,
        aspect_ratio=span ** 2 / area,
        taper=tip / root if root > 0 else 0.0,
        sweep_le_deg=_fit_angle(eta[meas], x_le[meas]),
        dihedral_deg=_fit_angle(eta[meas], h[meas]) if axis == 1 else 0.0,
        x_le_root_m=_extrapolate(eta, x_le, root_eta, d),
        x_ac_m=x_le_mac + 0.25 * mac,
        z_m=float((chord * z_col).sum() * d / area),
        airfoil=naca_name(thickness[inner_m], camber[inner_m], camber_pos[inner_m]),
        thickness_ratio=float(np.median(thickness)),
        camber=float(np.median(camber)),
        stations=[(float(a), float(b), float(c), float(e)) for a, b, c, e in zip(pos, chord, x_le, z_col)],
        part_ids=sorted(p.id for p in parts),
        sources={k: "cad" for k in ("span_m", "area_m2", "mac_m", "root_chord_m", "tip_chord_m", "aspect_ratio",
                                    "taper", "sweep_le_deg", "dihedral_deg", "x_le_root_m", "x_ac_m", "z_m",
                                    "thickness_ratio", "camber")} | {"airfoil": "inferred"},
    )


def _control_surface(part: Part) -> ControlSurfaceGeom | None:
    axis = 2 if part.role == "rudder" else 1
    prof = _profile(part.mesh, axis, N_CS_STATIONS)
    if prof is None:
        return None
    span = float(prof["hi"] - prof["lo"])
    area = float(prof["chord"].sum() * prof["d"])
    return ControlSurfaceGeom(id=part.id, role=part.role, area_m2=area, chord_m=area / span, span_m=span,
                              parent_surface=CONTROL_PARENT[part.role])


# --------------------------------------------------------------------------- geometry

def _is_fuselage(p: Part) -> bool:
    return p.type in FUSELAGE_TYPES or p.role == "fuselage"


def _aero_parts(design: Design) -> list[Part]:
    roles = WING_ROLES + HTAIL_ROLES + VTAIL_ROLES + VEE_ROLES + ("nacelle",)
    return [p for p in design.parts.values() if p.mesh is not None and (p.role in roles or _is_fuselage(p))]


def _cache_key(parts: list[Part], centre: float) -> str:
    h = hashlib.sha1(repr(round(centre, 9)).encode())
    for p in sorted(parts, key=lambda q: q.id):
        v = p.mesh.vertices
        h.update(repr((p.id, p.type, p.role, len(v), len(p.mesh.faces), round(float(v.sum()), 9))).encode())
    return h.hexdigest()


def _centreline(design: Design) -> float:
    fus = [p for p in design.parts.values() if p.mesh is not None and _is_fuselage(p)]
    meshed = fus or [p for p in design.parts.values() if p.mesh is not None]
    if not meshed:
        return 0.0
    b = np.vstack([p.bbox_m for p in meshed])
    return float((b[:, 1].min() + b[:, 1].max()) / 2)


def _measure_aero(parts: list[Part], centre: float) -> AeroGeometry:
    geom = AeroGeometry()

    def by_role(roles: tuple[str, ...]) -> list[Part]:
        return [p for p in parts if p.role in roles]

    def area(ps: list[Part]) -> float:
        return float(sum(p.mesh.area for p in ps))

    wing, htail, vtail, vee = by_role(WING_ROLES), by_role(HTAIL_ROLES), by_role(VTAIL_ROLES), by_role(VEE_ROLES)

    fus = [p for p in parts if _is_fuselage(p)]
    fus_y = (min(p.bbox_m[0, 1] for p in fus), max(p.bbox_m[1, 1] for p in fus)) if fus else None
    geom.wing = _measure_surface("wing", wing, 1, centre, fuselage=fus_y) if wing else None
    if vee:
        geom.htail = _measure_surface("htail", vee + htail, 1, centre, fuselage=fus_y)
        spans_both = min(p.bbox_m[0, 1] for p in vee) < centre < max(p.bbox_m[1, 1] for p in vee)
        geom.vtail = _measure_surface("vtail", vee + vtail, 2, centre, panels=2 if spans_both else 1)
    else:
        geom.htail = _measure_surface("htail", htail, 1, centre, fuselage=fus_y) if htail else None
        geom.vtail = _measure_surface("vtail", vtail, 2, centre) if vtail else None
    for label, surf in (("wing", geom.wing), ("horizontal tail", geom.htail), ("vertical tail", geom.vtail)):
        if surf is None:
            geom.warnings.append(f"no {label} found (by part role)")

    for p in parts:
        if p.role in CONTROL_PARENT:
            cs = _control_surface(p)
            if cs is None:
                geom.warnings.append(f"{p.id}: control surface could not be sliced")
            else:
                geom.control_surfaces.append(cs)

    if fus:
        b = np.vstack([p.bbox_m for p in fus])
        ext = b.max(axis=0) - b.min(axis=0)
        geom.fuselage_length_m, geom.fuselage_max_diam_m = float(ext[0]), float(max(ext[1], ext[2]))
    else:
        geom.warnings.append("no fuselage found (type fuselage/hatch or role fuselage)")

    geom.wetted_area_m2 = {"wing": area(wing), "htail": area(htail + vee), "vtail": area(vtail),
                           "fuselage": area(fus), "nacelles": area(by_role(("nacelle",)))}
    return geom


def _external_other_area(design: Design, aero: list[Part]) -> float:
    """Surface area of non-aero meshed parts that are not enclosed by an aero part's bbox (1 mm slack)."""
    ids = {p.id for p in aero}
    boxes = np.array([p.bbox_m for p in aero]) if aero else np.zeros((0, 2, 3))
    total = 0.0
    for p in design.parts.values():
        if p.mesh is None or p.id in ids or p.type == "prop":
            continue
        inside = ((boxes[:, 0] - 1e-3 <= p.bbox_m[0]) & (p.bbox_m[1] <= boxes[:, 1] + 1e-3)).all(axis=1)
        if not inside.any():
            total += float(p.mesh.area)
    return total


def extract_geometry(design: Design) -> AeroGeometry:
    """Measure wing/tail planforms, control surfaces, fuselage size and wetted areas from the meshes."""
    aero = _aero_parts(design)
    centre = _centreline(design)
    key = _cache_key(aero, centre)
    if key in _cache:
        _cache.move_to_end(key)
    else:
        _cache[key] = _measure_aero(aero, centre)
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    geom = copy.deepcopy(_cache[key])
    geom.wetted_area_m2["other"] = _external_other_area(design, aero)
    return geom


# --------------------------------------------------------------------------- mesh builders

def naca4(code: str, n: int = 60) -> np.ndarray:
    """Closed NACA 4-digit ring (2n-2, 2) in chord units: LE, upper surface to TE, lower surface back.

    Uses the closed-trailing-edge thickness coefficient (-0.1036) so the TE is a single point.
    """
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:4]) / 100
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0:
        fwd = x < p
        yc = np.where(fwd, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dyc = np.where(fwd, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    th = np.arctan(dyc)
    upper = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lower = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    return np.vstack([upper[:-1], lower[:0:-1]])   # LE, upper..., TE, lower... (excl. LE again)


def _loft(sections: list[tuple[float, float, float, float, str]], axis: int, n: int) -> trimesh.Trimesh:
    """Watertight loft through airfoil sections (span position, chord, x_le, offset, NACA code).

    ``axis`` 1: sections in planes of constant y, offset = z of the chord line. ``axis`` 2: planes of
    constant z (a fin), offset = y of the chord line.
    """
    k = 2 * n - 2
    rings = []
    for s, c, x_le, off, code in sections:
        af = naca4(code, n)
        ring = np.empty((k, 3))
        ring[:, 0] = x_le + af[:, 0] * c
        ring[:, axis] = s
        ring[:, 3 - axis] = off + af[:, 1] * c
        rings.append(ring)
    faces = []
    for i in range(len(rings) - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            faces += [(a, b, b + k), (a, b + k, a + k)]
    # end caps: strips between upper point i (ring index i) and lower point i (ring index (k - i) % k)
    for base in (0, (len(rings) - 1) * k):
        for i in range(n - 1):
            if i + 1 < n - 1:
                faces.append((base + i, base + i + 1, base + k - i - 1))
            if i > 0:
                faces.append((base + i, base + k - i - 1, base + k - i))
    mesh = trimesh.Trimesh(np.vstack(rings), np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def make_naca4_panel(y0: float, y1: float, chord0: float, chord1: float, airfoil0: str = "2412",
                     airfoil1: str | None = None, x_le0: float = 0.0, x_le1: float | None = None,
                     z0: float = 0.0, z1: float | None = None, n: int = 60) -> trimesh.Trimesh:
    """One straight wing panel between spanwise stations y0 and y1 (sections in planes of constant y)."""
    return _loft([(y0, chord0, x_le0, z0, airfoil0),
                  (y1, chord1, x_le0 if x_le1 is None else x_le1, z0 if z1 is None else z1, airfoil1 or airfoil0)],
                 axis=1, n=n)


def make_naca4_wing(span: float, root_chord: float, tip_chord: float, airfoil: str = "2412",
                    sweep_deg: float = 0.0, dihedral_deg: float = 0.0, x_le: float = 0.0, z: float = 0.0,
                    y_centre: float = 0.0, tip_airfoil: str | None = None, n: int = 60) -> trimesh.Trimesh:
    """Symmetric straight-tapered wing (one watertight body) with LE sweep and dihedral, root at y_centre."""
    half = span / 2
    dx, dz = half * np.tan(np.radians(sweep_deg)), half * np.tan(np.radians(dihedral_deg))
    tip = tip_airfoil or airfoil
    return _loft([(y_centre - half, tip_chord, x_le + dx, z + dz, tip),
                  (y_centre, root_chord, x_le, z, airfoil),
                  (y_centre + half, tip_chord, x_le + dx, z + dz, tip)], axis=1, n=n)


def make_naca4_fin(height: float, root_chord: float, tip_chord: float, airfoil: str = "0010",
                   sweep_deg: float = 0.0, x_le: float = 0.0, z0: float = 0.0, y: float = 0.0,
                   n: int = 60) -> trimesh.Trimesh:
    """Vertical fin from z0 up to z0 + height, sections in planes of constant z."""
    dx = height * np.tan(np.radians(sweep_deg))
    return _loft([(z0, root_chord, x_le, y, airfoil), (z0 + height, tip_chord, x_le + dx, y, airfoil)],
                 axis=2, n=n)


def make_fuselage(length: float, diam: float, x_nose: float = 0.0, y: float = 0.0, z: float = 0.0,
                  sections: int = 48) -> trimesh.Trimesh:
    """Streamlined body of revolution along +x: elliptic nose (25 %), constant section, tapered tail cone."""
    t = np.concatenate([np.linspace(0, 0.25, 12), np.linspace(0.3, 0.6, 4), np.linspace(0.65, 1.0, 8)])
    r = np.where(t < 0.25, np.sqrt(np.clip(1 - ((0.25 - t) / 0.25) ** 2, 0, 1)), 1.0)
    r = np.where(t > 0.6, 1 - 0.85 * (t - 0.6) / 0.4, r) * diam / 2
    r[0] = r[-1] = 0.0
    mesh = trimesh.creation.revolve(np.column_stack([r, t * length]), sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))   # z -> x
    mesh.apply_translation([x_nose, y, z])
    mesh.fix_normals()
    return mesh


def make_tube(od: float, id: float, length: float, axis: str = "y",
              center: tuple[float, float, float] = (0.0, 0.0, 0.0), sections: int = 48) -> trimesh.Trimesh:
    """Round tube (``id`` 0 -> solid rod) of ``length`` along ``axis`` ('x', 'y' or 'z'), centred at ``center``."""
    if id > 0:
        mesh = trimesh.creation.annulus(r_min=id / 2, r_max=od / 2, height=length, sections=sections)
    else:
        mesh = trimesh.creation.cylinder(radius=od / 2, height=length, sections=sections)
    rot = {"x": ([0, 1, 0], np.pi / 2), "y": ([1, 0, 0], -np.pi / 2), "z": ([0, 0, 1], 0.0)}[axis]
    mesh.apply_transform(trimesh.transformations.rotation_matrix(rot[1], rot[0]))
    mesh.apply_translation(center)
    return mesh
