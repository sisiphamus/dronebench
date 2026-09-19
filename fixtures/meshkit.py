"""Watertight mesh builders for the fixture generators.

Everything is built in METRES in the DroneBench frame (x aft, y right, z up; nose at small x)
and written to STL in millimetres by `save_mm`. No boolean engine is needed: every body is
built from explicit, index-shared vertex rings, so trimesh reports it watertight.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import trimesh

Airfoil = Union[str, tuple]   # "4411" or (m, p, t) as fractions of chord


# ---------------------------------------------------------------- airfoils
def naca_params(code: Airfoil) -> tuple[float, float, float]:
    if isinstance(code, str):
        c = code.upper().replace("NACA", "").strip()
        return int(c[0]) / 100.0, int(c[1]) / 10.0, int(c[2:4]) / 100.0
    return tuple(float(v) for v in code)


def blend_airfoil(a: Airfoil, b: Airfoil, f: float) -> tuple[float, float, float]:
    pa, pb = np.array(naca_params(a)), np.array(naca_params(b))
    return tuple(pa + (pb - pa) * f)


def naca4_surfaces(code: Airfoil, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower surface points ((n,2) each, columns x,z) at chord fractions x (open TE, ~0.25% thick)."""
    m, p, t = naca_params(code)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    if m > 0 and p > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    th = np.arctan(dyc)
    up = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lo = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    # keep stations at identical x so the cap ladder is valid (the sin(th) shift is < 0.3% chord)
    up[:, 0] = x
    lo[:, 0] = x
    return up, lo


def chord_stations(f0: float, f1: float, n: int) -> np.ndarray:
    """Cosine-clustered chord fractions between f0 and f1 (dense at the LE when f0 == 0)."""
    s = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    return f0 + (f1 - f0) * s


def naca4_coords(code: Airfoil, n: int = 60) -> np.ndarray:
    """Closed airfoil outline (TE upper -> LE -> TE lower), (2n-1, 2), chord 1."""
    x = chord_stations(0, 1, n)
    up, lo = naca4_surfaces(code, x)
    return np.vstack([up[::-1], lo[1:]])


# ---------------------------------------------------------------- generic lofts
def _ring_faces(k: int, a0: int, b0: int) -> list:
    f = []
    for i in range(k):
        j = (i + 1) % k
        f.append([a0 + i, a0 + j, b0 + j])
        f.append([a0 + i, b0 + j, b0 + i])
    return f


def loft(rings: Sequence[np.ndarray], cap: Sequence[Sequence[int]] | None = None) -> trimesh.Trimesh:
    """Closed solid through rings (each (k,3), same k). `cap` = triangles (ring-local indices) used to
    close both ends; default is a fan from the ring centroid (fine for convex rings)."""
    rings = [np.asarray(r, float) for r in rings]
    k = len(rings[0])
    verts = np.vstack(rings)
    faces = []
    for s in range(len(rings) - 1):
        faces += _ring_faces(k, s * k, (s + 1) * k)
    n = len(verts)
    if cap is None:
        c0, c1 = rings[0].mean(0), rings[-1].mean(0)
        verts = np.vstack([verts, c0, c1])
        for i in range(k):
            j = (i + 1) % k
            faces.append([n, j, i])
            faces.append([n + 1, (len(rings) - 1) * k + i, (len(rings) - 1) * k + j])
    else:
        last = (len(rings) - 1) * k
        for t in cap:
            faces.append([t[2], t[1], t[0]])
            faces.append([last + t[0], last + t[1], last + t[2]])
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    _orient(mesh)
    return mesh


def tube_loft(outer: Sequence[np.ndarray], inner: Sequence[np.ndarray]) -> trimesh.Trimesh:
    """Hollow shell: outer rings and inner rings (same count, same k); ends closed by annuli."""
    ns, k = len(outer), len(outer[0])
    verts = np.vstack([*outer, *inner])
    off = ns * k
    faces = []
    for s in range(ns - 1):
        faces += _ring_faces(k, s * k, (s + 1) * k)
        faces += [[f[0], f[2], f[1]] for f in _ring_faces(k, off + s * k, off + (s + 1) * k)]
    # annular caps
    faces += [[f[0], f[2], f[1]] for f in _ring_faces(k, 0, off)]
    faces += _ring_faces(k, (ns - 1) * k, off + (ns - 1) * k)
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    _orient(mesh)
    return mesh


def _orient(mesh: trimesh.Trimesh):
    trimesh.repair.fix_winding(mesh)
    if mesh.volume < 0:
        mesh.invert()


# ---------------------------------------------------------------- wings and surfaces
def airfoil_ring(code: Airfoil, f0: float, f1: float, n: int):
    """Section ring (chord 1) between chord fractions f0..f1 and its ladder cap triangles."""
    x = chord_stations(f0, f1, n) if f0 == 0 else np.linspace(f0, f1, n)
    up, lo = naca4_surfaces(code, x)
    if f0 == 0:
        ring = np.vstack([up, lo[1:][::-1]])          # u0..u(n-1), l(n-1)..l1   (u0 == l0)
        L = lambda i: 0 if i == 0 else 2 * n - 1 - i
    else:
        ring = np.vstack([up, lo[::-1]])              # u0..u(n-1), l(n-1)..l0
        L = lambda i: 2 * n - 1 - i
    cap = []
    for i in range(n - 1):
        u0, u1, l0, l1 = i, i + 1, L(i), L(i + 1)
        cap.append([u0, u1, l1])
        if u0 != l0:
            cap.append([u0, l1, l0])
    return ring, cap


def loft_wing_panel(y0, y1, chord0, chord1, x_le0, x_le1, z0, z1, airfoil0: Airfoil = "2412",
                    airfoil1: Airfoil | None = None, n=60, f0=0.0, f1=1.0, axis="y") -> trimesh.Trimesh:
    """Closed wing (or surface) panel between spanwise stations y0 and y1.

    Only the chord portion f0..f1 is built: a fixed panel with a control surface uses f1 = 1 - cf,
    the control surface f0 = 1 - cf, f1 = 1. axis="z" builds a vertical fin (span along +z, y0/y1 are
    heights, z0/z1 are the lateral y offset)."""
    airfoil1 = airfoil0 if airfoil1 is None else airfoil1
    rings, cap = [], None
    for y, c, xle, z, af in ((y0, chord0, x_le0, z0, airfoil0), (y1, chord1, x_le1, z1, airfoil1)):
        r2, cap = airfoil_ring(af, f0, f1, n)
        xs = xle + r2[:, 0] * c
        th = r2[:, 1] * c
        if axis == "y":
            rings.append(np.column_stack([xs, np.full(len(xs), y), z + th]))
        else:
            rings.append(np.column_stack([xs, z + th, np.full(len(xs), y)]))
    return loft(rings, cap)


# ---------------------------------------------------------------- fuselage
def superellipse(w: float, h: float, e: float = 2.5, k: int = 48) -> np.ndarray:
    """(k,2) points (y,z) of a superellipse |y/a|^e + |z/b|^e = 1 with full width w, height h."""
    t = np.linspace(0, 2 * math.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    y = (w / 2) * np.sign(c) * np.abs(c) ** (2 / e)
    z = (h / 2) * np.sign(s) * np.abs(s) ** (2 / e)
    return np.column_stack([y, z])


def fuselage_shell(stations: Sequence[tuple[float, float, float, float]], wall: float = 0.0015,
                   e: float = 2.5, k: int = 48) -> trimesh.Trimesh:
    """Hollow printed fuselage section through stations (x, width, height, z_centre), wall thickness `wall`."""
    outer, inner = [], []
    for x, w, h, zc in stations:
        o = superellipse(w, h, e, k)
        i = superellipse(max(w - 2 * wall, 1e-4), max(h - 2 * wall, 1e-4), e, k)
        outer.append(np.column_stack([np.full(k, x), o[:, 0], o[:, 1] + zc]))
        inner.append(np.column_stack([np.full(k, x), i[:, 0], i[:, 1] + zc]))
    return tube_loft(outer, inner)


def solid_body(stations: Sequence[tuple[float, float, float, float]], e: float = 2.5, k: int = 36,
               y: float = 0.0) -> trimesh.Trimesh:
    """Solid superellipse loft (nacelles, fairings, spinners) along x."""
    rings = []
    for x, w, h, zc in stations:
        o = superellipse(w, h, e, k)
        rings.append(np.column_stack([np.full(k, x), o[:, 0] + y, o[:, 1] + zc]))
    return loft(rings)


# ---------------------------------------------------------------- primitives
def box(size, center) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    return m


def _align(m: trimesh.Trimesh, axis: str, center) -> trimesh.Trimesh:
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    return m


def cylinder(radius, length, center, axis="x", sections=32) -> trimesh.Trimesh:
    return _align(trimesh.creation.cylinder(radius=radius, height=length, sections=sections), axis, center)


def make_tube(outer_d, inner_d, length, center, axis="y", sections=32) -> trimesh.Trimesh:
    """Round tube (spar, boom). Diameters and length in metres."""
    m = trimesh.creation.annulus(r_min=inner_d / 2, r_max=outer_d / 2, height=length, sections=sections)
    return _align(m, axis, center)


def motor(center, diam=0.035, length=0.032, shaft_len=0.012) -> trimesh.Trimesh:
    """Outrunner: can + shaft stub pointing forward (-x). `center` = can centre."""
    can = cylinder(diam / 2, length, center, "x")
    shaft = cylinder(0.0025, shaft_len, (center[0] - length / 2 - shaft_len / 2 + 1e-4, center[1], center[2]), "x",
                     16)
    return trimesh.util.concatenate([can, shaft])


def prop(center, diam, pitch, hub_d=0.018, hub_len=0.012, blade_chord_frac=0.09, n=12, spin=1,
         blade_angle_deg=90.0) -> trimesh.Trimesh:
    """Two-blade twisted prop in the y-z plane (axis along x). Blade pitch angle
    beta(r) = atan(pitch / (2 pi r)). `blade_angle_deg` rotates the blade pair about the axis."""
    R = diam / 2
    parts = [cylinder(hub_d / 2, hub_len, center, "x", 24)]
    for sgn in (1, -1):
        rings = []
        for r in np.linspace(hub_d * 0.35, R, n):
            beta = math.atan2(pitch, 2 * math.pi * r) * spin * sgn
            frac = r / R
            chord = diam * blade_chord_frac * (0.75 + 0.9 * frac * (1 - frac)) * (1 - 0.6 * frac**4)
            thick = max(0.0015, 0.012 * (1 - frac) + 0.001)
            sec = np.array([[-thick / 2, -chord / 2], [thick / 2, -chord / 2], [thick / 2, chord / 2],
                            [-thick / 2, chord / 2]])  # (x, tangential)
            cb, sb = math.cos(beta), math.sin(beta)
            rot = sec @ np.array([[cb, sb], [-sb, cb]]).T   # chord tilted from the disc plane by beta
            rings.append(np.column_stack([rot[:, 0], rot[:, 1], np.full(4, sgn * r)]))
        blade = loft(rings)
        blade.apply_transform(trimesh.transformations.rotation_matrix(math.radians(blade_angle_deg - 90), [1, 0, 0]))
        blade.apply_translation(center)
        parts.append(blade)
    return trimesh.util.concatenate(parts)


def servo(center, dims=(0.023, 0.012, 0.024), orient="z") -> trimesh.Trimesh:
    """Micro servo: case box plus mounting-tab plate. dims = (length, width, height); orient = height axis."""
    L, W, H = dims
    if orient == "z":
        body, tab = (L, W, H), (L + 0.009, W, 0.0025)
        toff = (0, 0, H * 0.25)
    elif orient == "flat":   # lying in a wing: height along x, length along y, width (thin side) along z
        body, tab = (H, L, W), (0.0025, L + 0.009, W)
        toff = (H * 0.25, 0, 0)
    elif orient == "y":
        body, tab = (L, H, W), (L + 0.009, 0.0025, W)
        toff = (0, H * 0.25, 0)
    else:
        body, tab = (H, W, L), (0.0025, W, L + 0.009)
        toff = (H * 0.25, 0, 0)
    c = np.asarray(center, float)
    return trimesh.util.concatenate([box(body, c), box(tab, c + np.array(toff))])


def save_mm(mesh: trimesh.Trimesh, path: Path):
    m = mesh.copy()
    m.apply_scale(1000.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def loft_surface(stations, f0=0.0, f1=1.0, n=40, axis="y") -> trimesh.Trimesh:
    """Closed surface through several stations [(span_pos, chord, x_le, offset, airfoil), ...].
    axis="y": span_pos = y, offset = z (wings, stabs). axis="z": span_pos = z, offset = y (fins)."""
    rings, cap = [], None
    for sp, c, xle, off, af in stations:
        r2, cap = airfoil_ring(af, f0, f1, n)
        xs, th = xle + r2[:, 0] * c, off + r2[:, 1] * c
        col = np.full(len(xs), sp)
        rings.append(np.column_stack([xs, col, th]) if axis == "y" else np.column_stack([xs, th, col]))
    return loft(rings, cap)


def rod(p0, p1, diam, sections=12) -> trimesh.Trimesh:
    """Solid round rod between two points (hinge wire, pushrod)."""
    return trimesh.creation.cylinder(radius=diam / 2, segment=[p0, p1], sections=sections)


# ---------------------------------------------------------------- fuselage profile helper
class Hull:
    """Fuselage outer profile: stations (x, width, height, z_centre) in metres, superellipse sections."""

    def __init__(self, stations, wall=0.0015, e=2.5):
        self.st = np.array(stations, float)
        self.wall, self.e = wall, e

    def at(self, x):
        s = self.st
        return tuple(float(np.interp(x, s[:, 0], s[:, i])) for i in (1, 2, 3))

    def top(self, x):
        w, h, zc = self.at(x)
        return zc + h / 2

    def bottom(self, x):
        w, h, zc = self.at(x)
        return zc - h / 2

    def floor(self, x0, x1, half_w, inner=True, n=12):
        """Highest hull (inner or outer) surface z at lateral offset +-half_w over x0..x1: an object whose
        flat bottom sits here touches the hull without cutting it."""
        zs = []
        for x in np.linspace(x0, x1, n):
            w, h, zc = self.at(x)
            a, b = w / 2 - (self.wall if inner else 0), h / 2 - (self.wall if inner else 0)
            r = min(abs(half_w) / a, 0.999)
            zs.append(zc - b * (1 - r ** self.e) ** (1 / self.e))
        return max(zs)

    def section(self, x0, x1, n=8, k=48) -> trimesh.Trimesh:
        xs = np.unique(np.concatenate([np.linspace(x0, x1, n), self.st[(self.st[:, 0] > x0) & (self.st[:, 0] < x1), 0]]))
        return fuselage_shell([(x, *self.at(x)) for x in xs], self.wall, self.e, k)


# ---------------------------------------------------------------- assembly -> design folder
try:
    import yaml as _yaml

    class _NoAlias(_yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True
except ImportError:  # pragma: no cover
    _NoAlias = None

class Assembly:
    """Collects parts (mesh in metres + BOM fields) and writes cad/*.stl + bom.yaml."""

    def __init__(self, scale: float = 1000.0):
        self.scale = scale   # mesh units -> file units (metres -> mm by default)
        self.parts: list[dict] = []
        self.meshes: dict[str, trimesh.Trimesh] = {}
        self.cg_hint: dict[str, np.ndarray] = {}   # CG (m) of parts whose CAD is not a mesh here (e.g. STEP)

    def add(self, id, type, mesh=None, cad_ext=".stl", **fields):
        entry = {"id": id, "type": type}
        if mesh is not None:
            entry["cad"] = id + cad_ext
            self.meshes[id] = mesh
        for k_, v in fields.items():
            if v is not None and v != [] and v != {}:
                entry[k_] = v
        self.parts.append(entry)
        return entry

    def get(self, id):
        return next(p for p in self.parts if p["id"] == id)

    def remove(self, id):
        self.parts = [p for p in self.parts if p["id"] != id]
        self.meshes.pop(id, None)

    def to_parts(self):
        """dronebench.model.Part objects (mesh in metres) for mass / CG estimates while building."""
        from dronebench.model import Material, Part
        out = []
        for e in self.parts:
            mat = None
            if "material" in e:
                md = e["material"]
                mat = Material(name=md["name"], density_kg_m3=md.get("density_kg_m3", 0) or 0,
                               kind=md.get("kind", ""), infill=md.get("infill"), walls=md.get("walls"),
                               orientation=md.get("orientation"), sub=md.get("sub", {}), source="bom")
            p = Part(id=e["id"], type=e["type"], part_number=e.get("part_number"), material=mat,
                     specs=dict(e.get("specs", {})), qty=e.get("qty", 1))
            m = self.meshes.get(e["id"])
            if m is not None:
                p.mesh, p.volume_m3, p.area_m2 = m, float(m.volume), float(m.area)
                p.cg_m = np.asarray(m.center_mass if m.is_volume else m.centroid)
            elif e["id"] in self.cg_hint:
                p.cg_m = self.cg_hint[e["id"]]
            elif "cg_mm" in p.specs:
                p.cg_m = np.asarray(p.specs["cg_mm"], float) / 1000.0
            out.append(p)
        return out

    def mass_cg(self, exclude=()):
        from dronebench.materials import part_mass_g
        tot, mom, rows = 0.0, np.zeros(3), []
        for p in self.to_parts():
            if p.id in exclude:
                continue
            g, src = part_mass_g(p)
            g *= p.qty
            cg = p.cg_m if p.cg_m is not None else np.zeros(3)
            tot += g
            mom += g * cg
            rows.append((p.id, g, src))
        return tot, mom / max(tot, 1e-9), rows

    def write(self, out_dir: Path, mission: dict, header: str = ""):
        import shutil
        import yaml
        out_dir = Path(out_dir)
        cad = out_dir / "cad"
        if cad.exists():
            shutil.rmtree(cad)
        cad.mkdir(parents=True)
        for id_, m in self.meshes.items():
            e = self.get(id_)
            if e["cad"].endswith((".stl", ".3mf")):
                mm_ = m.copy()
                mm_.apply_scale(self.scale)
                mm_.export(cad / e["cad"])
        with open(out_dir / "bom.yaml", "w") as f:
            if header:
                f.write("".join(f"# {ln}\n" for ln in header.strip().splitlines()))
            yaml.dump({"parts": self.parts}, f, Dumper=_NoAlias, sort_keys=False, default_flow_style=None, width=120)
        with open(out_dir / "mission.yaml", "w") as f:
            yaml.safe_dump(mission, f, sort_keys=False, default_flow_style=None)
