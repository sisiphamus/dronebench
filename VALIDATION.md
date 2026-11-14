# What DroneBench can and cannot be trusted for

This page lists what was checked, against what, and what is still open. Numbers are from the Falcon V2 replica in
`fixtures/falcon_v2` unless stated. Every figure here can be regenerated (commands at the end).

## What was wrong with the first version

The first DroneBench scored a design with closed-form formulas whose constants had been fitted to one aircraft's
manufacturer figure (1.6 Wh/km). Checked against independent tools, it was off in ways its stated 15% error band hid:

| Quantity | First version | Independent estimate | Comment |
|---|---|---|---|
| Best lift-to-drag at cruise | 12.7 | 17.8 (AeroSandbox section polars + body build-up) | 40% apart; no way to say which was right without a reference |
| Maximum lift coefficient | 1.15 | 1.50 | affects stall speed by 14% |
| Neutral point (static margin) | 0.491 m (16.9%) | 0.498 m (19.8%) vortex lattice + fuselage term | agreed to 7 mm |
| Neutral point from AeroSandbox's build-up alone | n/a | 0.525 m (31%) | wrong: it ignores the wing's downwash on the tail. Not used for stability. |
| Dynamics | none | none | a design was never flown, so stability, trim drag, gust loads and control authority were not tested at all |

It also scored a design by what you could delete: removing a middle fuselage section, the rudder, or all the wiring raised the score,
because the model had no way to see that the result could not fly. Those loopholes are closed (see the gate), but
the root cause was that nothing was simulated.

## What it does now

1. **Aerodynamics from geometry** (`dronebench/aero.py`, AeroSandbox). Section polars (NeuralFoil) and a body build-up give lift,
   drag and pitching moment against angle of attack through stall, at three airspeeds so Reynolds-number effects are real. The vortex lattice gives the
   stability and damping derivatives. Control effectiveness comes from the build-up with surfaces deflected. A slender-body term
   adds the fuselage's destabilising moment. Which tool is used for what was decided by testing each against the others (docstring).
2. **A 6-degree-of-freedom simulator** (`flight.py`, `_dyn.py`): rigid body, quaternion attitude, RK4, numba-compiled. Motor, propeller and
   battery with voltage sag and state of charge. Dryden turbulence. Servos with lag, rate and travel limits. Inertia from the part masses.
3. **A stock autopilot whose gains come from each design's own derivatives** (`autopilot.py`), so every design gets the same
   controller design method. A design the controller cannot fly is a design problem.
4. **Flown missions** (`mission.py`): a full-throttle climb, orbits at two states of charge and upwind/downwind legs, each in calm air, light wind
   with turbulence, and strong wind with turbulence. Endurance, range, energy per km and climb are measured from these flights.
   Static margin, stall margin, peak load factor, stalls and failure to complete are gate checks.

## Checks that pass today

Physics tests (`tests/test_flight_physics.py`, run on every change):

- trimmed flight is steady (altitude within 3 m, airspeed within 0.6 m/s over 6 s);
- control signs: positive aileron rolls right, positive elevator pitches up, positive rudder yaws right; a crosswind pushes the aircraft downwind;
- steady roll rate after an aileron input equals the value predicted by the roll damping and aileron derivatives (within 20%);
- unpowered phugoid period within 0.8 to 1.5 times Lanchester's pi * sqrt(2) * V / g;
- energy balance in a 40 s unpowered glide: loss of kinetic + potential energy equals the work done by drag to within 5%
  (checks the integrator, the frame conversions and the table lookups against each other);
- steady battery power in the simulator equals the trim-solver estimate within 12%;
- Dryden gust standard deviations match the specified values (sigma_u about 1.06 m/s, sigma_w about 0.7 m/s).

System tests: an unstable design (negative static margin) fails the static margin check; a battery moved 25 cm aft fails; the same
airframe with a battery move of 2 cm reuses the aerodynamic table (no rebuild); the autopilot flies 100% of randomised wind and
disturbance episodes where doing nothing from a disturbed start does not.

## One external anchor, and what it is worth

Titan Dynamics states 1.6 Wh/km for the Falcon V2 at 3.5 kg, cruising at 45 to 65 km/h. DroneBench was not tuned to it (`cd0_factor` is
1.0). Flown in calm air at 3.43 kg:

| km/h | trim estimate Wh/km | flown Wh/km |
|---|---|---|
| 45 | 1.56 | 1.57 |
| 55 | 1.64 | 1.65 |
| 65 | 1.85 | 1.87 |

This is a manufacturer claim, not an independent flight test, and it is one aircraft at one mass. It shows the model is in the
right place without tuning. It does not show the model is right for a different airframe. Reproduce with `validation/falcon_claim.py`.

## Open problems (read before trusting a result)

- **Only one real data point.** Nothing here has been compared with measured flight logs. `dronebench calibrate` fits the single
  drag factor to measured battery power in steady flight and reports the residual; with real logs this is the first thing to do.
- **Propeller model is a generic fit**, calibrated against published APC 11x8E curves, not per-prop measured data. Thrust and
  efficiency for unusual props are less certain than for common ones.
- **Prop slipstream over the wing, aeroelastic wing bending, ground effect, takeoff and landing are not modelled.** Missions start in the air.
- **Conventional layout only.** The aerodynamic builder needs a wing, a horizontal tail and a vertical tail. V-tails, flying wings and VTOL
  are not yet supported and fail with a clear message instead of a made-up result.
- **Structure is a spar, a gate on printed shells, and a load-factor limit.** Skins and joints are not analysed. Shells need two walls (a skin and
  printed ribs); that is a manufacturing rule, not a stress result.
- **Fixed weather.** Every design flies the same three conditions, so scores compare. A design tuned to those conditions could overfit them;
  `mission.CONDITIONS` can be replaced, and the benchmark runner can be given held-out seeds.
- **Stall and post-stall** come from the section polars and a table extrapolation above 25 degrees. Treat anything past the stall angle as qualitative.
- **Closed-form numbers are still produced** for the part-by-part removal test (it needs a fast model). They rank parts; the flown run decides.

## Regenerate

```
.venv/bin/python -m pytest -q tests/test_flight_physics.py tests/test_flown.py tests/test_envs.py
.venv/bin/python validation/falcon_claim.py
.venv/bin/dronebench score fixtures/falcon_v2 --flown --summary
```
