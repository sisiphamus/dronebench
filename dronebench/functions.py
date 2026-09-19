"""Rule-based part function + a one-line "why", with specifics from the part graph.

`describe(part, design, graph)` never returns an empty string.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .assemble import CONTROL_ROLES, LinkIndex, pair_motors, part_cg, part_mass_g, servo_surface
from .model import Design, Part

AVIONICS = ("fc", "rx", "gps", "telemetry", "vtx", "remote_id", "sensor", "camera")
_TYPE_FUNCTION = {
    "motor": "propulsion", "prop": "propulsion", "esc": "propulsion", "battery": "power", "pdb": "power",
    "wire": "power", "connector": "power", "servo": "control", "linkage": "control", "horn": "control",
    "hinge": "control", "fc": "avionics", "rx": "avionics", "gps": "avionics", "telemetry": "avionics",
    "vtx": "avionics", "remote_id": "avionics", "sensor": "avionics", "camera": "payload",
    "spar": "structure", "fuselage": "structure", "mount": "structure", "hatch": "structure",
    "fastener": "fastening", "ballast": "ballast", "payload": "payload", "wing": "lift",
    "tail": "stability", "surface": "control",
}


def _names(ids, limit=4) -> str:
    ids = list(ids)
    s = ", ".join(ids[:limit])
    return s + (f" +{len(ids) - limit} more" if len(ids) > limit else "")


def duplicate_of(part: Part, design: Design, links: LinkIndex) -> Optional[str]:
    """Another part with the same type + part_number whose links cover all of this part's links."""
    if part.type not in AVIONICS or part.locked:
        return None
    conn = ("electrical", "mechanical")
    mine = set(links.nbrs(part.id, conn))
    for o in design.parts.values():
        if o.id == part.id or o.type != part.type or (o.part_number or "") != (part.part_number or ""):
            continue
        theirs = set(links.nbrs(o.id, conn))
        # the earlier-listed / shorter-named one is the original
        if (mine - {o.id}) <= theirs and (len(o.id), o.id) < (len(part.id), part.id):
            return o.id
    return None


def _side(part: Part) -> str:
    c = part_cg(part)
    if c is None or abs(c[1]) < 0.01:
        return "centre"
    return "left" if c[1] < 0 else "right"


def _wing_why(part: Part, design: Design, links: LinkIndex, role: str, word: str) -> str:
    side = _side(part)
    same = [p for p in design.parts.values() if p.role == role and p.type == part.type and _side(p) == side]
    same.sort(key=lambda p: abs(part_cg(p)[1]) if part_cg(p) is not None else 0)
    bits = [f"{word} panel {[p.id for p in same].index(part.id) + 1} of {len(same)} ({side})"
            if len(same) > 1 else f"{word} ({side})" if side != "centre" else word]
    spars = links.nbrs(part.id, types=("spar",))
    if spars:
        bits.append("carries " + _names(spars))
    surfs = [n for n in links.nbrs(part.id) if design.parts[n].role in CONTROL_ROLES]
    surfs += [p.id for p in design.parts.values() if p.parent == part.id and p.role in CONTROL_ROLES and p.id not in surfs]
    if surfs:
        bits.append("hinge for " + _names(surfs))
    return "; ".join(bits)


def describe(part: Part, design: Design, graph=None, links: Optional[LinkIndex] = None) -> tuple[str, str]:
    links = links or LinkIndex(design, graph)
    t, pid = part.type, part.id
    nb = links.nbrs(pid)

    if part.locked or pid in design.mission.payload_ids:
        return "payload", f"mission payload ({t}); locked, cannot be removed"

    dup = duplicate_of(part, design, links)
    if dup:
        return _TYPE_FUNCTION.get(t, "avionics"), f"duplicate of {dup}; no unique connection"

    if t == "motor":
        pr = pair_motors(design, links).get(pid, {})
        n = len(design.of_type("motor"))
        bits = [f"drives {pr['prop']} via shaft" if pr.get("prop") else "no prop attached",
                f"powered by {pr['esc']}" if pr.get("esc") else "no ESC linked",
                f"1 of {n} thrust sources" if n > 1 else "only thrust source"]
        return "propulsion", "; ".join(bits)
    if t == "prop":
        motor = next((m for m, v in pair_motors(design, links).items() if v.get("prop") == pid), None)
        return "propulsion", f"turns motor torque into thrust; on {motor}" if motor else "prop with no motor linked"
    if t == "esc":
        motors = links.nbrs(pid, ("electrical",), types=("motor",))
        batt = links.path_to(pid, ("battery",))
        return "propulsion", (f"drives {_names(motors) or 'no motor'}; fed by {batt or 'no battery'}")
    if t == "battery":
        loads = links.nbrs(pid, ("electrical",))
        n = len(design.of_type("battery"))
        why = f"energy store; feeds {_names(loads) or 'nothing linked'}"
        return "power", why + (f"; 1 of {n} packs" if n > 1 else "")
    if t == "servo":
        surf = servo_surface(part, links)
        fc = links.path_to(pid, ("fc",), through=("wire", "connector", "pdb", "rx"))
        return "control", (f"moves {surf}" if surf else "drives no control surface") + \
            (f"; signal from {fc}" if fc else "; no signal link to FC")
    if t in ("linkage", "horn", "hinge"):
        return "control", f"{t} in the control run: {_names(nb) or 'unlinked'}"
    if t == "fc":
        outs = links.nbrs(pid, ("electrical",))
        power = [n for n in outs if design.parts[n].type in ("battery", "pdb", "wire", "connector")]
        cmds = [n for n in outs if n not in power]
        return "avionics", f"flight controller; commands {_names(cmds, 6) or 'nothing linked'}" + (
            f"; powered via {_names(power)}" if power else "")
    if t == "rx":
        return "avionics", f"radio receiver; pilot link to {_names(links.nbrs(pid, ('electrical',))) or 'nothing linked'}"
    if t in ("gps", "telemetry", "vtx", "remote_id", "sensor"):
        label = {"gps": "position fix", "telemetry": "ground-station data link", "vtx": "video transmitter",
                 "remote_id": "broadcast Remote ID", "sensor": "sensor input"}[t]
        return "avionics", f"{label} for {_names(links.nbrs(pid, ('electrical',))) or 'nothing linked'}"
    if t == "camera":
        return "payload", "camera (not locked as mission payload)"
    if t == "spar":
        carried = [n for n in nb if design.parts[n].type in ("wing", "tail", "fuselage")]
        what = "tail boom load" if part.role == "tail_boom" else "wing bending load"
        return "structure", f"carries {what} across {_names(carried, 5) or 'no linked panel'}"
    if t == "wing" or part.role == "wing":
        return "lift", _wing_why(part, design, links, part.role or "wing", "wing")
    if part.role in CONTROL_ROLES:
        driver = next((s.id for s in design.of_type("servo") if servo_surface(s, links) == pid), None)
        return "control", f"{part.role} on {part.parent or 'airframe'}; " + (
            f"driven by {driver}" if driver else "no servo")
    if t == "tail" or part.role in ("htail", "vtail", "vtail_ruddervator"):
        kind = {"htail": "pitch stability (horizontal tail)", "vtail": "yaw stability (fin)",
                "vtail_ruddervator": "pitch and yaw stability (V-tail)"}.get(part.role, "stability")
        return "stability", f"{kind}; " + _wing_why(part, design, links, part.role or "htail", "tail").split("; ", 1)[-1]
    if t == "fuselage":
        return "structure", f"fuselage; houses {_names([n for n in nb if design.parts[n].type not in ('fastener',)], 6) or 'nothing linked'}"
    if t == "mount":
        held = [n for n in nb if design.parts[n].type not in ("fastener",)]
        return "structure", f"mount; holds {_names(held) or 'nothing linked'}"
    if t == "hatch":
        return "structure", f"access hatch; closes {_names(nb) or 'the fuselage'}"
    if t == "fastener":
        from .necessity import fastened_parts
        held = sorted(fastened_parts(design, pid, links))
        size = part.specs.get("size") or part.part_number or "fastener"
        grp = f" (group {part.group})" if part.group else ""
        if len(held) >= 2:
            where = f"holding {held[0]} to {_names(held[1:])}"
        elif held:
            where = f"holding {held[0]}"
        else:
            where = "not touching any part"
        return "fastening", f"{part.qty}× {size} {where}{grp}"
    if t == "ballast":
        c = part_cg(part)
        at = f" at x={c[0]:.3f} m" if c is not None else ""
        cgx = None
        try:
            cgs = [(part_mass_g(p), part_cg(p)) for p in design.parts.values() if part_cg(p) is not None]
            tot = sum(m for m, _ in cgs)
            cgx = sum(m * c_[0] for m, c_ in cgs) / tot if tot > 0 else None
        except Exception:
            pass
        direction = ""
        if c is not None and cgx is not None:
            direction = " to move CG forward" if c[0] < cgx else " to move CG aft"
        return "ballast", f"adds {part_mass_g(part):.0f} g{at}{direction}"
    if t == "fairing":
        if nb:
            return "aerodynamic_fairing", f"fairing over {_names(nb)}; smooths flow, carries no load"
        return "cosmetic", "fairing linked to nothing structural; carries no load"
    if t == "payload":
        return "payload", "payload item (not locked)"
    if t in ("pdb", "wire", "connector"):
        return "power", f"{t}; links {_names(links.nbrs(pid, ('electrical',))) or 'nothing (no connects in BOM)'}"
    fn = _TYPE_FUNCTION.get(t, "unknown")
    return fn, f"{t} part; linked to {_names(nb) or 'nothing'}" + (
        f" (identified from CAD, confidence {part.confidence:.1f})" if part.confidence < 1 else "")
