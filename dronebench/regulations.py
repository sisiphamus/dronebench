"""Regulatory notes (US FAA/FCC, UN transport, EU market). Short strings, each cites its rule.

These are flags for a human, not legal advice.
"""
from __future__ import annotations

from typing import Optional

from .model import Design, Part, PhysicsResult

TRANSMITTERS = ("rx", "telemetry", "vtx", "remote_id")
ELECTRONICS = ("motor", "esc", "fc", "rx", "gps", "telemetry", "vtx", "camera", "remote_id", "pdb", "sensor",
               "servo", "battery", "connector", "wire")
CELL_V = {"lipo": 3.7, "liion": 3.6}


def pack_wh(part: Part) -> Optional[float]:
    s = part.specs
    if s.get("energy_wh"):
        return float(s["energy_wh"])
    try:
        from .assemble import battery_spec
        b = battery_spec(part)
    except Exception:
        b = None
    if b is None:
        return None
    return b.cells_s * CELL_V.get(b.chemistry, 3.7) * b.capacity_mah / 1000


def _truthy(v) -> bool:
    return v is True or str(v).lower() in ("true", "yes", "1")


def has_remote_id(design: Design) -> bool:
    if design.of_type("remote_id"):
        return True
    return any(_truthy(p.specs.get("has_remote_id")) for p in design.of_type("fc", "gps"))


def aircraft_regulations(design: Design, pres: Optional[PhysicsResult] = None, mass_kg: Optional[float] = None) -> list[str]:
    out = []
    m = mass_kg if mass_kg is not None else (pres.mass_kg if pres is not None else 0.0)
    if m >= 25:
        out.append(f"FAA: {m:.1f} kg ≥ 25 kg: Part 107 does not apply; needs a 49 USC 44807 exemption or type certificate")
    if m >= 0.25:
        out.append(f"FAA Part 107: {m * 1000:.0f} g ≥ 250 g: registration required (14 CFR 48); "
                   "remote pilot certificate for commercial use (14 CFR 107.12)")
        if has_remote_id(design):
            out.append("FAA Remote ID: broadcast module present (14 CFR Part 89)")
        else:
            out.append("MISSING: Remote ID module required (14 CFR Part 89)")
    else:
        out.append(f"FAA: {m * 1000:.0f} g < 250 g: registration not required for recreational flight (14 CFR 48.15)")
    packs = [(b.id, pack_wh(b)) for b in design.of_type("battery")]
    for bid, wh in packs:
        if wh is None:
            continue
        if wh > 160:
            out.append(f"{bid}: {wh:.0f} Wh > 160 Wh: prohibited on passenger aircraft (IATA DGR PI 965-967); "
                       "ships as UN3480/UN3481 Class 9")
        elif wh > 100:
            out.append(f"{bid}: {wh:.0f} Wh > 100 Wh: UN3480/UN3481 Class 9 dangerous goods for shipping; "
                       "airline carry-on only with approval (IATA DGR 2.3)")
    if packs:
        out.append("Lithium cells: UN38.3 test summary required for transport (UN Manual of Tests §38.3)")
    if any(p.type in TRANSMITTERS for p in design.parts.values()):
        out.append("Radio: every transmitter must carry an FCC ID (47 CFR 2.1033); "
                   "amateur-band links need a licensed operator (47 CFR Part 97)")
    return out


def part_regulations(part: Part, design: Optional[Design] = None, pres: Optional[PhysicsResult] = None) -> list[str]:
    t = part.type
    out = []
    transmits = t in TRANSMITTERS or (t == "gps" and _truthy(part.specs.get("transmits")))
    if transmits:
        out.append("FCC Part 15/97: device must carry an FCC ID (47 CFR 2.1033)")
    if t == "vtx":
        out.append("VTX: 5.8 GHz above 25 mW needs an amateur licence (47 CFR 97.215); EU limit 25 mW EIRP (ETSI EN 300 440)")
    if t == "remote_id":
        out.append("Remote ID: module must be on the FAA declaration-of-compliance list (14 CFR 89.510)")
    if t == "battery":
        wh = pack_wh(part)
        out.append("UN38.3 test summary required (UN Manual of Tests §38.3)")
        if wh is not None and wh > 160:
            out.append(f"{wh:.0f} Wh > 160 Wh: no airline carriage (IATA DGR); UN3480 Class 9 shipping")
        elif wh is not None and wh > 100:
            out.append(f"{wh:.0f} Wh > 100 Wh: UN3480/UN3481 Class 9 shipping restrictions")
    if t in ELECTRONICS:
        out.append("EU: RoHS 2011/65/EU compliance + REACH SVHC declaration (EC 1907/2006)")
    return out
