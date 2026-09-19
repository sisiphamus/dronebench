"""Self-contained HTML viewer: three.js model colored by necessity, part reports, score curve.

Axis mapping. Our frame is x aft, y right, z up (right-handed). glTF / three.js is
y up. We rotate -90 deg about x before export:

    three (X, Y, Z) = ours (x, z, -y)

so the nose points to -X (left on screen from the default camera), up is +Y, and the
right wing points to -Z (away from a camera on +Z, i.e. we look at the left side).
This is a proper rotation (det = +1), so nothing is mirrored.

Everything is embedded in one HTML file (GLB as base64, reports as JSON); three.js r160
is loaded from jsDelivr via an importmap, so the file works from file:// with no server.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

TEMPLATE = Path(__file__).with_name("template.html")
MAX_FACES = 200_000

# ours -> three.js
OURS_TO_THREE = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=float)


def _parts_of(design) -> list:
    parts = getattr(design, "parts", None)
    if parts is None:
        return []
    return list(parts.values()) if isinstance(parts, dict) else list(parts)


def _as_trimesh(mesh):
    import trimesh
    if mesh is None:
        return None
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    return mesh


def _decimate(meshes: dict, max_faces: int = MAX_FACES) -> tuple[dict, list[str]]:
    """Shrink meshes proportionally if the total face count is above max_faces. Never raises."""
    total = sum(len(m.faces) for m in meshes.values())
    notes = []
    if total <= max_faces:
        return meshes, notes
    ratio = max_faces / total
    out = {}
    for pid, m in meshes.items():
        target = max(64, int(len(m.faces) * ratio))
        if len(m.faces) <= target:
            out[pid] = m
            continue
        try:
            d = m.simplify_quadric_decimation(face_count=target)
            out[pid] = d if d is not None and len(d.faces) > 0 else m
        except Exception as e:  # fast_simplification missing, or it failed on this mesh
            out[pid] = m
            if not notes:
                notes.append(f"mesh decimation skipped ({type(e).__name__}); {total} faces embedded")
    return out, notes


def design_glb(design, max_faces: int = MAX_FACES) -> tuple[bytes, list[str], list[str]]:
    """GLB bytes of the design (one node per part named by part id, three.js frame).

    Returns (glb, part ids with a mesh, notes)."""
    import trimesh
    meshes = {}
    for p in _parts_of(design):
        try:
            m = _as_trimesh(getattr(p, "mesh", None))
        except Exception:
            m = None
        if m is not None:
            meshes[str(p.id)] = m
    meshes, notes = _decimate(meshes, max_faces)
    scene = trimesh.Scene()
    for pid, m in meshes.items():
        m = m.copy()
        m.apply_transform(OURS_TO_THREE)
        m.visual = trimesh.visual.ColorVisuals(m)  # drop textures; the viewer colors parts itself
        _ = m.vertex_normals
        scene.add_geometry(m, node_name=pid, geom_name=pid)
    if not meshes:
        return b"", [], notes + ["no part meshes to show"]
    return scene.export(file_type="glb"), list(meshes), notes


def _clean(o: Any) -> Any:
    """JSON-safe copy: NaN/inf -> None, numpy -> python, Paths -> str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _script_json(obj) -> str:
    # Safe inside <script>: no "</script>" or "<!--" can appear.
    return json.dumps(_clean(obj), separators=(",", ":")).replace("<", "\\u003c")


def render_viewer(design_dirs: Optional[list], reports: list[dict], out_html,
                  runs_jsonl=None, designs: Optional[list] = None,
                  labels: Optional[list[str]] = None, title: Optional[str] = None) -> Path:
    """Write one self-contained HTML viewer and return its path.

    design_dirs: design folders (loaded with ingest.load_design), or None if `designs` is given.
    designs: preloaded design objects (anything with .parts of objects having .id and .mesh).
    reports: one report dict (PLAN §4) per design. Two designs -> before/after view.
    runs_jsonl: optional runs file for the score-vs-step chart.
    """
    if designs is None:
        from dronebench.ingest import load_design
        designs = [load_design(Path(d)) for d in (design_dirs or [])]
    designs = list(designs)
    reports = list(reports)
    if len(designs) != len(reports):
        raise ValueError(f"got {len(designs)} designs but {len(reports)} reports")
    if not designs:
        raise ValueError("render_viewer needs at least one design")
    if labels is None:
        labels = ["before", "after"] if len(designs) == 2 else [""] * len(designs)

    runs = []
    if runs_jsonl:
        from dronebench.runs import load_runs
        runs = load_runs(runs_jsonl)

    glb_tags, entries = [], []
    for i, (d, rep) in enumerate(zip(designs, reports)):
        glb, ids, notes = design_glb(d)
        glb_tags.append(f'<script type="application/octet-stream" id="glb-{i}">'
                        f'{base64.b64encode(glb).decode("ascii")}</script>')
        name = rep.get("design") or getattr(d, "name", None) or (
            Path(design_dirs[i]).name if design_dirs else f"design {i + 1}")
        entries.append({"name": name, "label": labels[i] if i < len(labels) else "",
                        "mesh_ids": ids, "notes": notes, "report": rep})

    data = {"designs": entries, "runs": runs,
            "title": title or " vs ".join(e["name"] for e in entries)}
    html = TEMPLATE.read_text()
    html = html.replace("__TITLE__", _html_escape(data["title"]))
    html = html.replace("<!--__GLB__-->", "\n".join(glb_tags))
    html = html.replace("/*__DATA__*/null", _script_json(data))
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def _html_escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
