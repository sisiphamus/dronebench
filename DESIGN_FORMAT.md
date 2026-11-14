# Design folder format

A design is a folder. Bring your own by writing one and running `dronebench init ws --fixture path/to/folder`.

```
my_design/
  cad/          one watertight mesh per part (STL, OBJ, GLB, 3MF or STEP)
  bom.yaml      the parts list
  mission.yaml  what the aircraft is for
  params.yaml   optional: knobs for a parametric builder (see below)
```

Frame: x aft from the nose, y right, z up. Other frames and units are declared in `mission.yaml` and converted on load.

## mission.yaml

```yaml
preset: endurance        # endurance | range | speed | payload | efficiency
payload_ids: [camera]    # parts that must stay (removing one fails the gate)
cruise_kmh: 55           # optional
max_mass_kg: 7.0         # optional takeoff limit; the gate fails above it
frame: {units: mm, forward: "-x", up: "+z"}
```

## bom.yaml

```yaml
parts:
  - id: wing_L1              # unique
    type: wing               # motor prop esc battery servo fc rx gps camera spar wing fuselage tail surface
                             # hinge linkage horn fastener wire ballast fairing payload other (full list: dronebench/model.py)
    role: wing               # wing htail vtail aileron flap elevator rudder fuselage nacelle; spars: main_spar, rear_spar
    cad: wing_L1.stl         # file in cad/; omit for wires and small electronics (give mass_g and cg_mm in specs)
    part_number: any text    # matched against the component catalog where it can be
    material: {name: LW-PLA, infill: 0.0, walls: 1, orientation: vase}
    specs: {}                # numbers the physics reads, e.g. motor: kv, rm_ohm, i0_a, imax_a, mass_g
    connects: [esc_L, prop_L]
    group: motor_L_mount     # fasteners with the same group are removed as one unit
    qty: 1
    locked: false
```

What the flown evaluation needs to find in the parts list: a main wing, a horizontal tail, a vertical tail (parts with those roles), at
least one motor with a prop and a battery with an ESC, and every part attached to the airframe (by touching or by `connects`).
A design with a missing piece fails with the reason instead of receiving a score. Conventional wing and tail layouts only; V-tails and
flying wings are not supported yet.

Specs that an agent types for a new part are tagged `agent_supplied` and checked for plausibility (battery Wh/kg, motor power per gram);
implausible values are clamped.

## params.yaml (optional)

A parametric design ships a builder that regenerates `cad/` and `bom.yaml` from these numbers, so an agent can run
`dronebench build ws --set wing.span_mm=2200` or start a workspace from its own numbers with `dronebench init ws --set wing.span_mm=1900`.
The builder is named by the `builder:` key (a file `fixtures/build_<name>.py` with `build(params, out_dir, verbose)`).
`fixtures/falcon_v2/params.yaml` is the worked example.

## What comes back

`dronebench score DESIGN --flown` writes a report: `score` (value, unit), `gate` (every check with value, limit and pass), `aircraft`
(mass, CG, span, area, static margin, stall, cruise, endurance, range, Wh/km, climb), `flight` (the flown segments and their measurements), and
`parts` (for every part: function, why it is there, whether it is needed, mass, material, margin). `dronebench status ws` prints the
part that matters most to you: the smallest gate margins and the parts the removal test calls unnecessary.
