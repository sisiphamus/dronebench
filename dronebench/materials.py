"""Material database, material inference and per-part mass.

Values are typical room-temperature figures for the material class, taken from
manufacturer datasheets and standard references (cited per line). They are meant
for first-order mass and strength estimates, not certification.

Public API
----------
MATERIALS: dict[str, Material]      canonical name -> Material (source "catalog")
resolve_material(part) -> Material  never empty; source bom | catalog | assumed
part_mass_g(part) -> (grams, source)  mass of ONE unit (qty is applied by geometry.mass_properties)
"""
from __future__ import annotations

import copy
import re
from typing import Optional

from .model import Material, Part

LINE_WIDTH_M = 0.45e-3          # FDM extrusion line width used for shell thickness
FOAM_FACTORS = {"LW-PLA": 0.45}  # active-foaming filament: printed density / bulk density


def _m(name, rho, kind, uts=None, e=None, **kw) -> Material:
    return Material(name=name, density_kg_m3=rho, kind=kind, tensile_mpa=uts, modulus_gpa=e,
                    source="catalog", **kw)


MATERIALS: dict[str, Material] = {m.name: m for m in [
    # --- FDM filaments (printed) ---
    _m("PLA", 1240, "printed", 50, 3.5),          # NatureWorks Ingeo 4043D datasheet: 1.24 g/cc, ~50 MPa, 3.5 GPa
    _m("LW-PLA", 1240, "printed", 25, 1.5),       # ColorFabb LW-PLA: bulk 1.24 g/cc, foams to ~0.54 g/cc at 230 C (factor 0.45)
    _m("PETG", 1270, "printed", 50, 2.1),         # Eastman Eastar 6763 / Prusament PETG: 1.27 g/cc, ~50 MPa, 2.1 GPa
    _m("ASA", 1070, "printed", 45, 2.0),          # Prusament ASA: 1.07 g/cc
    _m("ABS", 1040, "printed", 40, 2.2),          # typical ABS: 1.04 g/cc
    _m("PA12", 1010, "printed", 48, 1.7),         # EOS PA2200 (SLS PA12): ~0.93-1.01 g/cc, 48 MPa, 1.7 GPa
    _m("PA-CF", 1150, "printed", 80, 6.0),        # Polymaker PA6-CF / Bambu PAHT-CF: ~1.15 g/cc, 80 MPa, ~6 GPa
    _m("TPU", 1210, "printed", 35, 0.03),         # TPU 95A: 1.21 g/cc
    # --- composites ---
    _m("CF tube", 1550, "composite", 600, 100),   # pultruded CF tube (DragonPlate/Easy Composites): ~1.55 g/cc, 600+ MPa flex, ~100 GPa
    _m("CF sheet", 1500, "composite", 600, 70),   # quasi-isotropic CF plate
    _m("Fiberglass", 1850, "composite", 300, 20), # G10/E-glass laminate
    _m("FR4", 1850, "composite", 310, 22),        # FR4 PCB laminate
    # --- metals ---
    _m("Al 6061", 2700, "metal", 240, 69),        # ASM: 6061-T6, 2.70 g/cc, yield 240 MPa (UTS 310), 68.9 GPa
    _m("Al 7075", 2810, "metal", 500, 71),        # ASM: 7075-T6, 2.81 g/cc, yield 503 MPa, 71.7 GPa
    _m("Steel", 7850, "metal", 400, 200),         # carbon steel (screws 8.8 class are higher; music wire ~2000)
    _m("Stainless steel", 8000, "metal", 505, 193),  # 304 SS
    _m("Brass", 8500, "metal", 340, 100),         # C360 brass
    _m("Copper", 8960, "metal", 210, 117),        # C110 copper
    _m("NdFeB", 7500, "metal", 80, 160),          # sintered NdFeB magnet, 7.4-7.6 g/cc (brittle)
    _m("Lead", 11340, "metal", 18, 16),
    # --- polymers (moulded) ---
    _m("Nylon PA66", 1140, "polymer", 80, 3.0),   # DuPont Zytel 101: 1.14 g/cc
    _m("Nylon glass-filled", 1350, "polymer", 150, 7.0),  # PA66-GF30 (APC "glass-filled nylon" props)
    _m("Polycarbonate", 1200, "polymer", 65, 2.4),  # Lexan: 1.20 g/cc
    _m("PVC", 1380, "polymer", 50, 3.0),          # battery shrink wrap
    _m("Silicone", 1100, "polymer", 8, 0.005),    # silicone wire insulation
    _m("Glass", 2500, "polymer", 50, 70),         # camera lens glass (grouped as polymer-like inert)
    # --- foams and woods ---
    _m("EPO foam", 33, "foam", 0.5, 0.01),        # EPO RC foam 30-35 kg/m3
    _m("EPP foam", 30, "foam", 0.3, 0.005),       # EPP 30 kg/m3 grade
    _m("Balsa", 160, "wood", 20, 3.5),            # medium balsa 160 kg/m3 (USDA Wood Handbook)
    _m("Plywood", 600, "wood", 40, 8.0),          # aircraft/lite ply
    # --- cells, electronics, mixed assemblies (effective densities) ---
    _m("Li-ion cell", 2450, "cell"),              # 21700 NMC (Molicel P42A: 70 g / 24.2 cm3 = ~2.9; incl. air in pack ~2.45)
    _m("LiPo pack", 2000, "cell"),                # soft-pouch LiPo pack, ~2.0 g/cc effective
    _m("Silicone wire", 3500, "mixed", sub={"conductor": "Copper", "insulation": "Silicone"}),  # 14-22 AWG effective
    _m("Electronics", 1500, "electronic", sub={"pcb": "FR4", "components": "mixed", "wires": "Copper+Silicone"}),
    _m("Motor assembly", 5000, "mixed", sub={"bell": "Al 6061", "shaft": "Steel", "windings": "Copper",
                                             "magnets": "NdFeB", "bearings": "Steel"}),
]}

ALIASES = {
    "pla": "PLA", "pla+": "PLA", "lw-pla": "LW-PLA", "lwpla": "LW-PLA", "lw pla": "LW-PLA",
    "lightweight pla": "LW-PLA", "foaming pla": "LW-PLA", "petg": "PETG", "pet-g": "PETG", "asa": "ASA", "abs": "ABS",
    "pa12": "PA12", "nylon 12": "PA12", "pa-cf": "PA-CF", "pacf": "PA-CF", "pa6-cf": "PA-CF", "nylon carbon": "PA-CF",
    "carbon nylon": "PA-CF", "paht-cf": "PA-CF", "tpu": "TPU",
    "carbon": "CF tube", "cf": "CF tube", "carbon fiber": "CF tube", "carbon fibre": "CF tube",
    "carbon fiber tube": "CF tube", "carbon fibre tube": "CF tube", "carbon tube": "CF tube", "cf tube": "CF tube",
    "cfrp": "CF tube", "cf rod": "CF tube", "carbon rod": "CF tube",
    "cf sheet": "CF sheet", "cf plate": "CF sheet", "carbon plate": "CF sheet", "carbon sheet": "CF sheet",
    "fiberglass": "Fiberglass", "fibreglass": "Fiberglass", "g10": "Fiberglass", "gfrp": "Fiberglass", "fr4": "FR4",
    "aluminium": "Al 6061", "aluminum": "Al 6061", "al": "Al 6061", "al6061": "Al 6061", "6061": "Al 6061",
    "al 6061": "Al 6061", "al7075": "Al 7075", "7075": "Al 7075", "al 7075": "Al 7075",
    "steel": "Steel", "music wire": "Steel", "piano wire": "Steel", "stainless": "Stainless steel",
    "stainless steel": "Stainless steel", "ss": "Stainless steel", "brass": "Brass", "copper": "Copper",
    "ndfeb": "NdFeB", "neodymium": "NdFeB", "lead": "Lead",
    "nylon": "Nylon PA66", "pa66": "Nylon PA66", "pa": "Nylon PA66", "nylon pa66": "Nylon PA66",
    "glass-filled nylon": "Nylon glass-filled", "glass filled nylon": "Nylon glass-filled",
    "nylon glass-filled": "Nylon glass-filled", "pa66-gf": "Nylon glass-filled", "pa66-gf30": "Nylon glass-filled",
    "polycarbonate": "Polycarbonate", "pc": "Polycarbonate", "lexan": "Polycarbonate", "pvc": "PVC",
    "silicone": "Silicone", "glass": "Glass",
    "epo": "EPO foam", "epo foam": "EPO foam", "epp": "EPP foam", "epp foam": "EPP foam", "foam": "EPO foam",
    "balsa": "Balsa", "plywood": "Plywood", "ply": "Plywood",
    "li-ion": "Li-ion cell", "liion": "Li-ion cell", "li-ion cell": "Li-ion cell", "21700": "Li-ion cell",
    "18650": "Li-ion cell", "lipo": "LiPo pack", "lipo pack": "LiPo pack",
    "silicone wire": "Silicone wire", "wire": "Silicone wire",
    "electronics": "Electronics", "electronic": "Electronics", "pcb": "Electronics",
    "motor": "Motor assembly", "motor assembly": "Motor assembly",
}

PRINTED_TYPES = {"wing", "tail", "fuselage", "surface", "hatch", "fairing", "mount", "horn"}
ELECTRONIC_TYPES = {"esc", "fc", "rx", "gps", "telemetry", "vtx", "remote_id", "pdb", "sensor", "connector"}

# per-unit fallback masses (g) when there is no specs mass, no mesh and no catalog entry
FALLBACK_MASS_G = {
    "motor": 100, "prop": 20, "esc": 30, "battery": 300, "servo": 12, "fc": 20, "rx": 5, "gps": 20,
    "telemetry": 15, "vtx": 10, "camera": 30, "remote_id": 12, "pdb": 15, "wire": 20, "connector": 5,
    "sensor": 10, "spar": 40, "wing": 200, "fuselage": 300, "tail": 50, "surface": 20, "hinge": 1,
    "linkage": 3, "horn": 1, "fastener": 1, "mount": 10, "hatch": 20, "ballast": 50, "fairing": 10,
    "payload": 100, "other": 10,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().replace("_", " "))


def get_material(name: Optional[str]) -> Optional[Material]:
    """Look up a material by name or alias (case-insensitive). Returns a copy, or None."""
    if not name:
        return None
    if name in MATERIALS:
        return copy.deepcopy(MATERIALS[name])
    key = _norm(name)
    for canon in MATERIALS:
        if _norm(canon) == key:
            return copy.deepcopy(MATERIALS[canon])
    if key in ALIASES:
        return copy.deepcopy(MATERIALS[ALIASES[key]])
    key2 = key.replace(" ", "").replace("-", "")
    for alias, canon in ALIASES.items():
        if alias.replace(" ", "").replace("-", "") == key2:
            return copy.deepcopy(MATERIALS[canon])
    return None


def foam_factor(mat: Material) -> float:
    """Printed-density / bulk-density factor. 0.45 for LW-PLA (foamed), else 1."""
    return FOAM_FACTORS.get(mat.name, 1.0)


def _printed(name="PLA", infill=0.15, walls=2, orientation="flat", source="assumed") -> Material:
    m = get_material(name)
    m.infill, m.walls, m.orientation, m.source = infill, walls, orientation, source
    return m


def _name_has(part: Part, *words: str) -> bool:
    text = " ".join(str(x) for x in (part.part_number, part.id, part.notes) if x).lower()
    return any(w.lower() in text for w in words)


def _infer_from_type(part: Part) -> Material:
    t = part.type
    if t in PRINTED_TYPES:
        name = "LW-PLA" if t in {"wing", "tail", "surface", "fairing"} and _name_has(part, "lw") else "PLA"
        if _name_has(part, "petg"):
            name = "PETG"
        return _printed(name, 0.15, 2, "flat")
    if t == "spar":
        if _name_has(part, "alu", "aluminium", "aluminum"):
            return _assumed("Al 6061")
        if _name_has(part, "wood", "dowel", "balsa"):
            return _assumed("Balsa")
        return _assumed("CF tube")
    if t == "motor":
        m = _assumed("Motor assembly")
        return m
    if t == "prop":
        if _name_has(part, "carbon", "cf "):
            return _assumed("CF sheet")
        return _assumed("Nylon glass-filled")  # APC thin-electric "E" props are glass-filled nylon
    if t == "servo":
        metal = _name_has(part, "mg", "ma ", "ma2", "metal") or re.search(r"\bma\b|\d+ma\b", str(part.part_number or "").lower())
        m = Material(name="Servo assembly", density_kg_m3=1800, kind="mixed",
                     sub={"case": "Nylon PA66", "gears": "Brass" if metal else "Nylon PA66",
                          "motor": "Copper+NdFeB", "pcb": "FR4"}, source="assumed")
        return m
    if t in ELECTRONIC_TYPES:
        return _assumed("Electronics")
    if t in {"camera", "payload"}:
        m = _assumed("Electronics")
        m.sub = {**m.sub, "lens": "Glass", "housing": "Al 6061" if _name_has(part, "gimbal") else "Polycarbonate"}
        return m
    if t == "battery":
        liion = _name_has(part, "li-ion", "liion", "p42a", "p45b", "21700", "18650", "molicel", "samsung", "vtc")
        m = _assumed("Li-ion cell" if liion else "LiPo pack")
        m.sub = {"cells": "Li-ion NMC 21700" if liion else "LiPo pouch", "casing": "PVC shrink",
                 "wires": "Copper+Silicone"}
        return m
    if t == "wire":
        return _assumed("Silicone wire")
    if t == "fastener":
        if _name_has(part, "nylon"):
            return _assumed("Nylon PA66")
        if _name_has(part, "alu", "aluminium", "aluminum"):
            return _assumed("Al 7075")
        return _assumed("Steel")
    if t == "hinge":
        if _name_has(part, "ca hinge", "ca-hinge"):
            m = Material(name="CA hinge", density_kg_m3=1400, kind="polymer", source="assumed")
            return m
        if _name_has(part, "rod", "pin", "wire"):
            return _assumed("Steel")
        return _assumed("Steel")
    if t == "linkage":
        return _assumed("Steel")  # music-wire pushrod
    if t == "ballast":
        return _assumed("Lead" if _name_has(part, "lead") else "Steel")
    return _printed("PLA", 0.15, 2, "flat")


def _assumed(name: str) -> Material:
    m = get_material(name)
    m.source = "assumed"
    return m


def resolve_material(part: Part) -> Material:
    """Material for a part. Priority: BOM material (merged with the catalog entry) > COTS catalog > part type.

    Never returns an empty material. `source` is bom, catalog or assumed.
    """
    bm = part.material
    if bm is not None and bm.name:
        base = get_material(bm.name)
        if base is None:
            base = copy.deepcopy(bm)
            if not base.density_kg_m3:
                inferred = _infer_from_type(part)
                base.density_kg_m3 = inferred.density_kg_m3
            if not base.kind or base.kind in ("unknown", "assumed"):
                base.kind = inferred.kind
        for f in ("infill", "walls", "orientation", "tensile_mpa", "modulus_gpa"):
            v = getattr(bm, f)
            if v is not None:
                setattr(base, f, v)
        if bm.density_kg_m3 and bm.density_kg_m3 > 0:
            base.density_kg_m3 = bm.density_kg_m3
        if bm.kind and bm.kind not in ("unknown", "assumed"):
            base.kind = bm.kind
        base.sub = {**base.sub, **(bm.sub or {})}
        if base.kind == "printed":
            base.infill = 0.15 if base.infill is None else base.infill
            base.walls = 2 if base.walls is None else base.walls
            base.orientation = base.orientation or "flat"
        base.source = "bom"
        return base
    if part.part_number:
        from .components import lookup
        cat = lookup(part.part_number)
        if cat and cat.get("material"):
            m = get_material(cat["material"]) or _infer_from_type(part)
            m.sub = {**m.sub, **cat.get("sub", {})}
            m.source = "catalog"
            return m
    return _infer_from_type(part)


def _vol_area(part: Part):
    vol, area = part.volume_m3, part.area_m2
    if part.mesh is not None:
        try:
            if vol is None:
                vol = abs(float(part.mesh.volume))
            if area is None:
                area = float(part.mesh.area)
        except Exception:
            pass
    return vol, area


def printed_mass_kg(volume_m3: float, area_m2: float, mat: Material) -> float:
    """FDM part mass from mesh volume and surface area.

    shell_fraction = min(1, area * walls * 0.45 mm / volume)
    rho_eff = rho * foam * (shell_fraction + (1 - shell_fraction) * infill)
    Vase mode (orientation 'vase', or walls <= 1 with infill 0): mass = area * walls * 0.45 mm * rho * foam
    (walls > 1 in vase mode stands for internal vase-printed ribs/spar webs).
    """
    rho = mat.density_kg_m3 * foam_factor(mat)
    walls = mat.walls if mat.walls is not None else 2
    infill = mat.infill if mat.infill is not None else 0.15
    vase = (mat.orientation or "").lower() == "vase" or (walls <= 1 and infill == 0)
    if vase:
        return area_m2 * max(walls, 1) * LINE_WIDTH_M * rho
    if volume_m3 <= 0:
        return area_m2 * walls * LINE_WIDTH_M * rho
    shell = min(1.0, area_m2 * walls * LINE_WIDTH_M / volume_m3)
    return volume_m3 * rho * (shell + (1.0 - shell) * infill)


def part_mass_g(part: Part) -> tuple[float, str]:
    """Mass of one unit of `part` in grams and its source tag. qty is NOT applied here.

    Priority:
      1. specs.mass_g (source "bom", or the part's existing agent_supplied* tag for mass_g)
      2. printed material + mesh: printed_mass_kg (source "cad")
      3. other material + mesh: volume * density (source "cad")
      4. COTS catalog mass by part_number (source "catalog")
      5. per-type fallback (source "assumed")
    """
    specs = part.specs or {}
    if specs.get("mass_g") is not None:
        try:
            m = float(specs["mass_g"])
            tag = part.sources.get("mass_g", "bom")
            if tag not in ("agent_supplied", "agent_supplied_clamped"):
                tag = "bom"
            if m > 0:
                return m, tag
        except (TypeError, ValueError):
            pass
    mat = resolve_material(part)
    vol, area = _vol_area(part)
    if vol is not None and area is not None and (vol > 0 or area > 0):
        if mat.kind == "printed":
            return printed_mass_kg(vol, area, mat) * 1000.0, "cad"
        if part.part_number:
            from .components import lookup
            cat = lookup(part.part_number)
            if cat and cat.get("mass_g"):
                return float(cat["mass_g"]), "catalog"
        if vol > 0:
            return vol * mat.density_kg_m3 * 1000.0, "cad"
    if part.part_number:
        from .components import lookup
        cat = lookup(part.part_number)
        if cat and cat.get("mass_g"):
            return float(cat["mass_g"]), "catalog"
    return float(FALLBACK_MASS_G.get(part.type, 10)), "assumed"
