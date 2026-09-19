"""Shared data contracts for DroneBench. Every module talks through these types.

Coordinate frame (after ingest normalisation), SI units unless a name says otherwise:
  x = aft (nose at small x), y = right wing, z = up. Metres.
CAD files are usually mm; ingest converts using Design.frame.

Every value that ends up in the report carries a source tag, one of SOURCES.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

SOURCES = ("bom", "cad", "computed", "inferred", "assumed", "catalog", "agent_supplied", "agent_supplied_clamped")

# Part types (bom.yaml `type`)
PART_TYPES = (
    "motor", "prop", "esc", "battery", "servo", "fc", "rx", "gps", "telemetry", "vtx", "camera", "remote_id",
    "pdb", "wire", "connector", "sensor",
    "spar", "wing", "fuselage", "tail", "surface", "hinge", "linkage", "horn", "fastener", "mount", "hatch",
    "ballast", "fairing", "payload", "other",
)

# Functions a part can serve (report `function`)
FUNCTIONS = ("lift", "stability", "control", "propulsion", "power", "avionics", "structure", "fastening",
             "payload", "aerodynamic_fairing", "ballast", "cosmetic", "unknown")

# Aero roles for lifting / control surfaces (bom.yaml `role`)
AERO_ROLES = ("wing", "htail", "vtail", "vtail_ruddervator", "aileron", "flap", "elevator", "rudder",
              "ruddervator", "elevon", "fuselage", "nacelle", None)


@dataclass
class Material:
    name: str                       # e.g. "LW-PLA", "PLA", "CF tube", "Al 6061"
    density_kg_m3: float            # bulk density of the solid material
    kind: str                       # printed | metal | composite | polymer | foam | wood | electronic | cell | mixed
    tensile_mpa: Optional[float] = None
    modulus_gpa: Optional[float] = None
    infill: Optional[float] = None  # printed parts only, 0-1
    walls: Optional[int] = None     # printed parts only
    orientation: Optional[str] = None  # printed parts only, e.g. "flat", "vertical", "vase"
    sub: dict[str, str] = field(default_factory=dict)  # sub-part materials for bought parts {"bell": "Al 7075", ...}
    source: str = "assumed"


@dataclass
class Part:
    id: str
    type: str                                   # one of PART_TYPES
    part_number: Optional[str] = None
    cad: Optional[str] = None                   # file (or body name) relative to design/cad
    role: Optional[str] = None                  # one of AERO_ROLES, or spar role "main_spar" / "rear_spar"
    material: Optional[Material] = None
    specs: dict[str, Any] = field(default_factory=dict)
    locked: bool = False                        # mission payload: removal = gate fail
    connects: list[str] = field(default_factory=list)   # explicit electrical/mechanical links (ids)
    group: Optional[str] = None                 # fastener group / joint name; removal test treats a group as one unit
    parent: Optional[str] = None                # e.g. aileron -> wing part it hinges on
    qty: int = 1
    notes: Optional[str] = None
    # filled by ingest / geometry
    mesh: Any = None                            # trimesh.Trimesh in metres, normalised frame (not serialised)
    volume_m3: Optional[float] = None
    area_m2: Optional[float] = None
    bbox_m: Optional[np.ndarray] = None         # (2,3) min,max
    mass_g: Optional[float] = None
    cg_m: Optional[np.ndarray] = None           # (3,)
    confidence: float = 1.0                     # identification confidence (1.0 = from BOM)
    sources: dict[str, str] = field(default_factory=dict)  # field -> source tag, e.g. {"mass_g": "cad"}


@dataclass
class Frame:
    units: str = "mm"         # mm | m | in
    forward: str = "-x"       # which CAD axis points to the nose
    up: str = "+z"


@dataclass
class Mission:
    preset: str = "endurance"               # endurance | range | speed | payload | efficiency
    payload_ids: list[str] = field(default_factory=list)
    cruise_kmh: Optional[float] = None       # optional required cruise; else best-range speed is used
    rho: float = 1.225
    usable_battery_fraction: float = 0.85


@dataclass
class Design:
    name: str
    root: Path                                # design folder
    parts: dict[str, Part] = field(default_factory=dict)
    mission: Mission = field(default_factory=Mission)
    frame: Frame = field(default_factory=Frame)
    warnings: list[str] = field(default_factory=list)

    def of_type(self, *types: str) -> list[Part]:
        return [p for p in self.parts.values() if p.type in types]

    def of_role(self, *roles: str) -> list[Part]:
        return [p for p in self.parts.values() if p.role in roles]


@dataclass
class SurfaceGeom:
    """One lifting surface (whole, both sides) measured from CAD."""
    name: str                     # wing | htail | vtail
    span_m: float                 # tip to tip (vtail: height)
    area_m2: float                # planform (projected) area incl. control surfaces
    mac_m: float
    root_chord_m: float
    tip_chord_m: float
    aspect_ratio: float
    taper: float
    sweep_le_deg: float
    dihedral_deg: float
    x_le_root_m: float
    x_ac_m: float                 # absolute x of aerodynamic centre (quarter MAC)
    z_m: float
    airfoil: Optional[str] = None       # e.g. "NACA4411" (root)
    thickness_ratio: float = 0.12
    camber: float = 0.0                 # max camber / chord
    stations: list[tuple[float, float, float, float]] = field(default_factory=list)  # (y, chord, x_le, z)
    part_ids: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)


@dataclass
class ControlSurfaceGeom:
    id: str
    role: str                      # aileron | flap | elevator | rudder | ruddervator | elevon
    area_m2: float
    chord_m: float                 # mean chord of the moving surface
    span_m: float
    parent_surface: str            # wing | htail | vtail


@dataclass
class AeroGeometry:
    wing: Optional[SurfaceGeom] = None
    htail: Optional[SurfaceGeom] = None
    vtail: Optional[SurfaceGeom] = None
    fuselage_length_m: float = 0.0
    fuselage_max_diam_m: float = 0.0
    wetted_area_m2: dict[str, float] = field(default_factory=dict)   # component -> wetted area
    control_surfaces: list[ControlSurfaceGeom] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MassProps:
    total_kg: float
    cg_m: np.ndarray               # (3,)
    by_part_g: dict[str, float] = field(default_factory=dict)


@dataclass
class MotorSpec:
    id: str
    kv: float
    rm_ohm: float
    i0_a: float
    imax_a: float
    mass_g: float = 0.0


@dataclass
class PropSpec:
    id: str
    diam_in: float
    pitch_in: float
    blades: int = 2


@dataclass
class BatterySpec:
    id: str
    cells_s: int
    capacity_mah: float
    c_rating: float                # continuous
    chemistry: str = "lipo"        # lipo | liion
    ir_mohm_per_cell: float = 5.0
    mass_g: float = 0.0
    parallel_packs: int = 1        # packs wired in parallel (capacity & current add)


@dataclass
class PowerTrain:
    """One motor + its prop + its ESC. Several trains share the battery."""
    motor: MotorSpec
    prop: PropSpec
    esc_imax_a: float


@dataclass
class SparSpec:
    id: str
    outer_d_mm: float
    inner_d_mm: float
    length_m: float                # length it spans in the wing (both sides if continuous)
    material: Material
    role: str = "main_spar"        # main_spar | rear_spar | tail_boom


@dataclass
class ServoSpec:
    id: str
    torque_kgcm: float
    surface_id: Optional[str]      # control surface it drives


@dataclass
class PhysicsInput:
    geom: AeroGeometry
    mass: MassProps
    trains: list[PowerTrain]
    battery: Optional[BatterySpec]
    spars: list[SparSpec]
    servos: list[ServoSpec]
    payload_kg: float = 0.0
    cruise_ms: Optional[float] = None   # required cruise speed if the mission sets one
    rho: float = 1.225
    usable_battery_fraction: float = 0.85
    load_factor: float = 3.5


@dataclass
class PhysicsResult:
    ok: bool                           # False if physics could not run (e.g. no wing); see notes
    mass_kg: float = 0.0
    stall_ms: float = float("nan")
    cruise_ms: float = float("nan")    # required cruise, else best-range speed
    best_endurance_ms: float = float("nan")
    best_range_ms: float = float("nan")
    vmax_ms: float = float("nan")
    climb_ms: float = float("nan")     # max rate of climb at current mass
    endurance_min: float = 0.0
    range_km: float = 0.0
    wh_per_km: float = float("nan")    # at cruise
    km_per_wh: float = 0.0
    cl_max: float = float("nan")
    cd0: float = float("nan")
    oswald_e: float = float("nan")
    ld_max: float = float("nan")
    neutral_point_x_m: float = float("nan")
    static_margin: float = float("nan")   # (x_np - x_cg)/MAC
    tail_vh: float = float("nan")
    tail_vv: float = float("nan")
    spar_sf: float = float("nan")
    motor_current_cruise_a: float = float("nan")   # per motor
    motor_current_max_a: float = float("nan")      # per motor, full throttle, static
    battery_current_max_a: float = float("nan")
    battery_c_max: float = float("nan")            # C rating needed at full throttle
    servo_margin: dict[str, float] = field(default_factory=dict)  # servo id -> torque / required
    battery_energy_wh: float = 0.0
    payload_fraction: float = 0.0
    error_band: float = 0.15
    notes: list[str] = field(default_factory=list)


@dataclass
class Check:
    name: str
    value: Any
    limit: Any
    passed: bool
    detail: str = ""


@dataclass
class GateResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


@dataclass
class NecessityResult:
    id: str                     # part id or group id
    verdict: str                # required | helpful | unnecessary
    reason: str
    score_delta: float = 0.0
    margins_before: dict[str, float] = field(default_factory=dict)
    margins_after: dict[str, float] = field(default_factory=dict)
