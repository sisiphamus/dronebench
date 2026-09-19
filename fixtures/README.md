# DroneBench fixtures

Design folders in the PLAN §3 format (`cad/`, `bom.yaml`, `mission.yaml`) plus the scripts that make them.

| Folder | Built by | Parts | AUW (part_mass_g) |
|---|---|---|---|
| `falcon_v2/` | `build_falcon_v2.py` (reads `falcon_v2/params.yaml`) | 80 | ~3.43 kg |
| `trainer/` | `build_trainer.py trainer` | 42 | ~1.03 kg |
| `bad_tail/` | `build_trainer.py bad_tail` | 42 | ~0.99 kg |
| `step_mix/` | `build_trainer.py step_mix` | 42 (4 STEP) | ~1.02 kg |
| `third_party/` | `fetch_third_party.py` (network) | 48 (21 meshes) | ~1.43 kg |

Rebuild everything: `.venv/bin/python fixtures/build_falcon_v2.py && .venv/bin/python fixtures/build_trainer.py && .venv/bin/python fixtures/fetch_third_party.py`.
Builds are deterministic; falcon_v2 rebuilds in < 1 s.

## Conventions
- Our own fixtures are written in mm in the DroneBench frame (x aft from the nose, y right, z up; `frame: {units: mm, forward: "-x", up: "+z"}`). Meshes are made by `meshkit.py` from index-shared vertex rings, so every STL is watertight.
- `specs.fits: [ids]` on a part = the parts it is allowed to sit inside (declared fits for the clearance check): spars in wing panels, servos in wings/stabs, hinge rods, pushrods, horns, the ESC in its nacelle, the spar doubler inside the main spar. Everything else interpenetrates < 0.3 mm (checked with signed distance). Parts that should touch are placed 0.1-0.3 mm apart.
- BOM parts with no CAD (fasteners, wiring, third-party electronics) carry `specs.mass_g` and `specs.cg_mm` (CAD-file frame and units).
- Fuselage sections are hollow printed shells; electronics sit on a printed/plywood tray that touches the fuselage floor.

## falcon_v2 (Titan Falcon V2 replica)
Manual figures: span 2093 mm, wing area 4514 cm², AR 9.71, NACA 4411 root / 3411 tip, root chord 260, tip chord 51, dihedral 0.5°, sweep 3°, stab NACA 2410, MTOW 7 kg, cruise 45-65 km/h, 1.6 Wh/km at 3.5 kg.

**Planform assumption** (the manual does not give the break): constant 260 mm chord to y = ±602 mm, then linear taper to 51 mm at the tip, 3° LE sweep on the tapered part only. S = 2·(0.260·0.602 + ½·(0.260+0.051)·0.4445) = 0.4513 m², MAC 235.2 mm, MAC LE at x = 382.8 mm, wing a.c. x = 441.6 mm. Root LE at x = 380 mm; the wing sits on top of the fuselage.

Per side: Wing1 (0-350 mm, full chord), Wing2 (350-700 mm, front 75 %) + flap (rear 25 %), Wing3 (700-1046.5 mm, front 75 %) + aileron (rear 25 %). LW-PLA vase mode, `walls: 2` = skin plus vase-printed internal ribs.

Tail: Hstab NACA 2410, S = 0.075 m², AR 4.5, taper 0.7, a.c. 750 mm behind the wing a.c. (Vh ≈ 0.53), split elevators (30 %). Vstab NACA 0010, S = 0.040 m², 220 mm tall (Vv ≈ 0.032), rudder 30 %. Fuselage: 5 printed LW-PLA shells (Fuse1-Fuse5, 2.4 mm wall).

Propulsion and systems: 2× BrotherHobby Avenger 2816 620KV in nacelles at y = ±300 mm, APC 11x8E tractor props (disc clears the fuselage by ~100 mm), 2× 35 A ESC (inside the nacelles), Matek F405-WTE, ELRS EP2, Matek M10Q GPS, Matek ASPD-4525 pitot, 7× Emax ES08MA II (2 aileron, 2 flap, 2 elevator, 1 rudder; each `connects` to its surface and the FC), hinge rods, horns, pushrods, fastener groups (`motor_L_mount`, `motor_R_mount`, `wing_mount`, `servo_mount`, `tail_mount`, `fuse_joints`). No Remote ID module (the real Falcon has none). Payload: Sony RX0 II camera in the nose, `locked: true`.

Spars: `main_spar` CF 10/8 mm, one straight 1600 mm tube at 25 % root chord; `rear_spar` CF 6/4 mm, 1200 mm at 58 % root chord. Straight tubes through a 0.5° dihedral wing stay inside the skin.

**CG choice**: battery (6S2P Molicel P42A, 900 g, rated 10C: the cell is 45 A continuous) placed automatically (`battery.x_mm: auto`) so the CG sits at `target_cg_frac_mac` = 0.29 MAC (x ≈ 451 mm). The pipeline gives SM ≈ 0.17 (NP ≈ 0.46 MAC). The nose ballast moves the CG forward by ≈ 0.04 MAC, so removing it brings SM to ≈ 0.13, closer to the middle of the 0.05-0.20 band; removing every seeded part leaves SM ≈ 0.09. The CG sits slightly forward on purpose so that the ballast and belly plate come back `unnecessary`, not `helpful`.

**Seeded waste** (switch off under `waste:` in params.yaml):
| id | what | why it is waste |
|---|---|---|
| `ballast_nose` | 80 g steel block in the nose | CG already in range without it |
| `gps_2` | second M10Q GPS on the tray | duplicate sensor |
| `spar_doubler` | 400 mm CF 8/6 sleeve inside the main spar root | main + rear spar already give SF ≥ 1.5 (see note) |
| `fairing_L/R` | wingtip pods | cosmetic |
| `fuse_armor` | 5 mm PETG belly plate, ~170 g | heavy protection, no function in the gate |

Spar note (pipeline figures): spar SF 2.20 with everything, 1.57 without the doubler (so the doubler comes back `helpful`, which is honest: it adds strength the gate does not need), 1.10 without `main_spar` (required).

Edit `falcon_v2/params.yaml` (span, chords, airfoils, tail areas/arm, spar sizes, motor/prop/ESC/servo/battery part numbers, CG target, waste switches) and rerun the script.

## trainer, bad_tail, step_mix
`trainer`: 1.2 m rectangular high wing (chord 200 mm, NACA 2412, 2° dihedral, S = 0.24 m²), SunnySky X2216 1100KV + APC 9x6E on a plywood firewall (10x7E drew 26.3 A at full throttle in the propulsion model, just over the 26 A motor rating), 30 A ESC, 3S 2200 mAh LiPo, CF 8/6 spar, Matek F405-WTE, ELRS EP2, M10Q GPS, 2× SG90 (ailerons) + 2× MG90S (elevator, rudder with long pushrods). Tail sized for Vh 0.45 / Vv 0.035 with a 560 mm arm. CG ≈ 0.29 MAC.

`bad_tail`: same airframe, tail areas × 0.4 (Vh ≈ 0.18, Vv ≈ 0.014) and the battery moved 300 mm aft into the tail boom (CG ≈ 0.46 MAC, behind the estimated NP). Must fail `tail_volume` and `static_margin`.

`step_mix`: the trainer with all four servos as STEP files (SG-90 model). Source: FreeCAD-library, `Electrical Parts/Servos/SG-90/Servo-sg90.step`, commit `be5e71491051cef75da46831b6fd77c6313cb497`, https://github.com/FreeCAD/FreeCAD-library, license CC-BY-3.0 (`LICENSE-Assets`; author per the repo history). The original is kept in `step_mix/source/`; each servo is a copy whose assembly placement (`#16/#17/#18`) is rewritten to put the part in place. cascadio loads them in mm at the right spot; they are not watertight, so mass comes from `specs.mass_g`.

## third_party (OpenGeoDrone)
Real open design: https://github.com/enacuavlab/OpenGeoDrone @ `4c611977679d`, GPL-3.0. A vase-mode VTOL flying wing (4 hover motors on two arms, 1 pusher, 2 elevons printed with the mid panels). The per-part STLs are ~190 MB, so the script downloads the ~8.4 MB Bambu 3MF plates, drops the plate transforms and re-assembles the parts spanwise from the SCAD parameters. The frame is kept as OpenSCAD's (mm, forward `-x`, up `+y`, z = left span) so ingest has to convert it. The wing panels are written as `.3mf` and the rest as `.stl`. Many meshes are not watertight (thin vase features). Electronics come from the README hardware list, with models and masses marked as assumed. Details in `third_party/SOURCE.md`. It has no tail and no separate control surfaces, so it is expected to fail the gate; it exists to test messy CAD.
