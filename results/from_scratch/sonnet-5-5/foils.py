"""Mesh helpers for the generator: lofted airfoil pieces that can be a chordwise slice (front or aft of a hinge line).
No boolean or cap-triangulation engine is installed, so every piece is built directly and closed with fan caps."""
import numpy as np
import trimesh

import sys
sys.path.insert(0, "../code")
from dronebench.geometry import naca4  # noqa: E402


def _curves(code, n=80):
    ring = naca4(code, n)
    up = ring[:n]                       # LE .. just before TE (index n-1 is TE)
    te = ring[n - 1]
    upper = np.vstack([ring[:n - 1], te])
    lower = np.vstack([ring[0], ring[n:][::-1], te]) if False else None
    low = np.vstack([ring[0:1], ring[n:][::-1], te[None, :]])  # LE, lower ascending, TE
    return upper, low


def section_ring(code, x0, x1, m=24):
    """Closed ring (chord units) of the airfoil between chord fractions x0..x1 (0..1)."""
    upper, lower = _curves(code)
    xs = x0 + (x1 - x0) * (1 - np.cos(np.linspace(0, np.pi, m))) / 2
    yu = np.interp(xs, upper[:, 0], upper[:, 1])
    yl = np.interp(xs, lower[:, 0], lower[:, 1])
    up = np.column_stack([xs, yu])
    lo = np.column_stack([xs, yl])[::-1]
    pts = np.vstack([up, lo])
    # drop duplicate points (closed LE/TE)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
            keep.append(i)
    pts = pts[keep]
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    return pts


def piece(axis, s0, s1, c0, c1, xle0, xle1, off0, off1, code, x0=0.0, x1=1.0, m=24):
    """Loft between two sections. axis 1: span along y, offset = z of chord line. axis 2: fin, span along z, offset = y."""
    ring = section_ring(code, x0, x1, m)
    k = len(ring)
    rings = []
    for s, c, xle, off in ((s0, c0, xle0, off0), (s1, c1, xle1, off1)):
        r = np.empty((k, 3))
        r[:, 0] = xle + ring[:, 0] * c
        r[:, axis] = s
        r[:, 3 - axis] = off + ring[:, 1] * c
        rings.append(r)
    verts = np.vstack(rings)
    faces = []
    for j in range(k):
        a, b = j, (j + 1) % k
        faces += [(a, b, b + k), (a, b + k, a + k)]
    for base, flip in ((0, False), (k, True)):
        cen = verts[base:base + k].mean(axis=0)
        ci = len(verts)
        verts = np.vstack([verts, cen])
        for j in range(k):
            a, b = base + j, base + (j + 1) % k
            faces.append((ci, b, a) if flip else (ci, a, b))
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def cat(meshes):
    return trimesh.util.concatenate(meshes)
