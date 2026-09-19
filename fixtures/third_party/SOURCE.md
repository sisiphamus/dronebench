# third_party: OpenGeoDrone

- Source: https://github.com/enacuavlab/OpenGeoDrone
- Commit: `4c611977679d9378b03034ca3fedfd96186c708e`
- License: GPL-3.0 (repository LICENSE). The meshes in `cad/` are derived from the repository's
  3MF print files and stay under GPL-3.0; keep this notice with them.
- Design: parametric vase-mode VTOL flying wing by ENAC UAV Lab (FlyHigh project, ANR-24-CE92-0015 /
  DFG GR 5407/2-1). Half-span 490 mm + 90 mm center body, root chord 210 mm, MH61 airfoil, 4 hover motors
  on two arms, 1 pusher, 2 elevons. README: mass without battery 1.066 kg, 4S 4 Ah battery.

## Files fetched (Bambu Studio X1C print plates, ~8.4 MB total)
- `3mf_print_part/X1C_Center_part.3mf`
- `3mf_print_part/X1C_Fuselage_and_Clamps.3mf`
- `3mf_print_part/X1C_Left_Mid_Aileron_part.3mf`
- `3mf_print_part/X1C_Left_Root_part.3mf`
- `3mf_print_part/X1C_Motor_Arms_and_sides.3mf`
- `3mf_print_part/X1C_Right_Mid_Aileron_part.3mf`
- `3mf_print_part/X1C_Right_Root_part.3mf`

The per-part STLs in `stl_print_part/` are ~190 MB, over our download budget, so the 3MF plates were used.
Each 3MF object is the part mesh (decimated by the slicer) re-centred on its bounding box; the plate
transforms (print orientation / bed position) were dropped, which keeps the original OpenSCAD axes.

## Re-assembly (approximate)
Translations were rebuilt in the OpenSCAD frame (mm, x aft, y up, z = left span):
- center part at the origin (symmetric about z = 0, width 90 = `center_width`);
- root panels stacked at |z| = 45..250 (`wing_root_mm` 215 - 10 mm hull), motor arms in the 38 mm gap
  (|z| = 250..288), mid+elevon panels at |z| = 288..535 (half-span 490 + 45);
- x/y of each panel matched to the neighbouring panel's airfoil section (bbox mid-point of the section at the
  joint; across the arm gap the root panel's LE/TE sweep is extrapolated);
- motor arm front/back halves meet at 40 % chord of the local section; fuselage front/bottom-back/upper are
  stacked as cut in `OpenGeoDrone.scad` (12 mm overlap) with the tail flush with the center part TE;
- 6 wing clamps placed at the joints; servo horns under the elevon roots.
Expect a few mm of error and some interpenetration (declared in `specs.fits`). The elevons are printed as
one body with the mid panels, so there is no separate control-surface part; as a tailless flying wing it
has no horizontal/vertical tail.

## Not in the CAD (from the README hardware list)
Hover motors (F40), 7x4 props, 4-in-1 ESC, pusher motor (< 1200 kV) + ESC + 9x6 prop, 4S 4 Ah battery,
Tawaki FC, RC receiver, GPS, telemetry modem, pitot, 2 servos, CF spars (5.5 / 6.5 mm), wiring, screws.
Their models, masses and positions are estimates (see `notes` in bom.yaml); `specs.cg_mm` is in the CAD frame.
