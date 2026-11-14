# What DroneBench can and cannot be trusted for

This page lists what was checked, against what, and what is still open. Numbers are from the Falcon V2 replica in
`fixtures/falcon_v2` unless stated. Every figure can be regenerated (commands at the end).

## What was wrong with the first version

The first DroneBench scored a design with closed-form formulas whose constants had been fitted to one aircraft's
manufacturer figure (1.6 Wh/km). It never flew a design, so stability, trim drag, gust loads and control authority were not tested at all.
Checked against independent tools:

| Quantity | First version | Independent estimate | Comment |
|---|---|---|---|
| Best lift-to-drag | 12.7 | 17.8 for the clean airframe in AeroSandbox; 13.4 for the full aircraft as flown | the clean airframe is far better than the real aircraft; motor pods, sensors and tail trim drag take it back down. The old figure was close for a reason the old model could not show. |
| Maximum lift coefficient | 1.15 | 1.50 | changes stall speed by 14% |
| Neutral point | 0.491 m (16.9% MAC) | 0.495 m (18.9%): vortex lattice + fuselage term, lift slope from the section polars | agrees to 5 mm |
| Neutral point from AeroSandbox's build-up alone | n/a | 0.525 m (31%) | wrong: it ignores the wing's downwash on the tail. Not used for stability. |

It also scored by what could be deleted: removing a middle fuselage section, the rudder, the wiring or the motor pods raised the score,
because the model had no way to see the result could not fly. An agent could also change `usable_battery_fraction` through the mission edit
op and raise its own endurance without touching the aircraft (found in review; closed: only `cruise_kmh` is editable now).

## What it does now

1. **Aerodynamics from geometry** (`dronebench/aero.py`, AeroSandbox). Section polars (NeuralFoil) and a body build-up give lift, drag and pitching
   moment against angle of attack through stall at three airspeeds, so Reynolds-number effects are real. The vortex lattice gives the stability
   and damping derivatives, and sets the pitching-moment slope below the stall. Control effectiveness comes from the build-up with surfaces deflected,
   with hinge lines from the real control-surface chord. A slender-body term adds the fuselage's destabilising moment. The choice of tool for each
   quantity was made by testing them against each other (docstring in `aero.py`).
2. **A 6-degree-of-freedom simulator** (`flight.py`, `_dyn.py`): rigid body, quaternion attitude, RK4, numba-compiled (15 microseconds per step).
   Motor, propeller and battery with voltage sag and state of charge. Dryden turbulence. Servos with lag, rate and travel limits. Inertia from the part masses.
3. **A stock autopilot whose gains come from each design's own derivatives** (`autopilot.py`), so every design gets the same controller
   design method. A design the controller cannot fly is a design problem.
4. **Flown missions** (`mission.py`): a full-throttle climb, orbits and upwind/downwind legs.
   *Scored* segments fly in calm air and in a steady 5 m/s wind (no random turbulence), so a design's score is identical every run and two
   designs differ only because the aircraft differs. *Robustness* segments fly in light and gusty turbulence and are pass/fail: does it still
   complete, within altitude and speed tracking limits, load factor and stall? Turbulence is not scored because a 30 to 130 s gust realisation
   moves average power by tens of percent through sustained updrafts (found in review). Endurance is usable energy divided by flown orbit power at two
   states of charge; range is usable energy divided by the mean of an upwind and a downwind leg's Wh per km.
5. **Gate checks that follow from flight and from physical sense**: static margin, stall margin, climb, every segment completes, peak load factor,
   no stall, a pack that fits its volume (`battery_fit`), the mission's takeoff mass limit (`mtow`, 7 kg for the Falcon V2 per its manual),
   plus attached parts, continuous fuselage, wiring present, printed shells with ribs, spar strength and current limits.

## Checks that pass today

Physics tests (`tests/test_flight_physics.py`, run on every change):

- trimmed flight is steady (altitude within 3 m, airspeed within 0.6 m/s over 6 s);
- control signs: positive aileron rolls right, positive elevator pitches up, positive rudder yaws right; a positive sideslip swings the nose to the right (weathercock);
- the compiled force and moment assembly (wind to body rotation, moment transfer to the CG, thrust moment) against an independent
  python implementation at 25 random flight states, to 1e-6;
- steady roll rate after an aileron input equals the value predicted by the roll damping and aileron derivatives (within 20%);
- unpowered phugoid period within 0.8 to 1.5 times Lanchester's pi * sqrt(2) * V / g;
- energy balance in a 40 s unpowered glide to 5% (checks the integrator and the quaternion kinematics; it reuses the compiled force function,
  which is why the independent assembly test above exists);
- steady battery power in the simulator equals the trim-solver estimate within 12%;
- Dryden gust standard deviations within 25% of the specified values (sigma_u about 1.06 m/s, sigma_w about 0.7 m/s).

System tests: an unstable design fails the static margin check; a battery moved 25 cm aft fails; a 3-inch propeller that cannot trim fails the gate
instead of passing with no flight checks; mass-only edits reuse the aerodynamic table; the score does not change when only the turbulence seeds
change; a steady wind raises energy per km; the design-aware autopilot completes all 12 randomised wind and disturbance episodes and scores
well above doing nothing.

Independent review: an agent that had not seen the work read all of it, ran 24 probes and reported the issues now fixed above
(mission edit exploit, turbulence noise in the score, wind cancelling in the range figure, a gate that passed an untrimmable aircraft, a pitching-moment
kink, tail-incidence drag, unused control-surface chord, weak tests, stale numbers, environment reward and reset problems).

## One external anchor, and what it is worth

Titan Dynamics states 1.6 Wh/km for the Falcon V2 at 3.5 kg, cruising at 45 to 65 km/h. DroneBench was not tuned to it (`cd0_factor` is 1.0).
Flown in calm air at 3.43 kg (`validation/falcon_claim.py`):

| km/h | trim estimate Wh/km | flown Wh/km |
|---|---|---|
| 45 | 1.46 | 1.47 |
| 55 | 1.52 | 1.53 |
| 65 | 1.78 | 1.79 |

The mean of the five speeds from 45 to 65 km/h is 1.58. This is a manufacturer claim, not an independent flight test, for one aircraft at one
mass. It shows the model is in the right place without tuning. It does not show the model is right for a different airframe, and one modelling
choice moves it: how much drag a built-in tail incidence costs (here 40% of what a deflected elevator would). Charging none gives about 1.43 at
55 km/h, charging the full amount about 1.64.

## Open problems (read before trusting a result)

- **Only one real data point.** Nothing here has been compared with measured flight logs. `dronebench calibrate` fits the single drag factor to
  measured battery power in steady flight and reports the residual; with real logs this is the first thing to do.
- **The propeller model is a generic fit**, calibrated against published APC 11x8E curves, not per-prop measured data. Thrust and efficiency for unusual
  props are less certain than for common ones. Idle (windmilling) prop drag and prop torque are not modelled.
- **Prop slipstream over the wing, aeroelastic wing bending, ground effect, takeoff and landing are not modelled.** Missions start in the air.
- **Conventional layout only.** The aerodynamic builder needs a wing, a horizontal tail and a vertical tail. V-tails, flying wings and VTOL are not
  supported and fail the gate with a message (`flight_model`) instead of receiving a made-up score.
- **Structure is a spar, a gate on printed shells, and a load-factor limit.** Skins and joints are not analysed; shells need two walls (a skin and printed ribs),
  a manufacturing rule and not a stress result. The flown load-factor check does not bind on its own (peaks are about 2 g against 7 to 16 g limits); the
  spar check at 3.5 g does the work.
- **Fixed weather.** Every design flies the same conditions, so scores compare. An optimiser could overfit the robustness seeds;
  `mission.evaluate_flight(..., seed_offset=n)` rotates them, so hold some back. The scored segments are deterministic and need no seeds.
- **Top speed is a trim calculation, not flown**, and is limited to 49 m/s (the edge of the propeller table).
- **Stall and post-stall** come from the section polars and a table extrapolation above 25 degrees. Treat anything past the stall angle as qualitative.
- **Closed-form numbers are still produced** for the part-by-part removal test (it needs a fast model). They rank parts; the flown run decides.
- **Part floating tolerance:** parts within 25 mm of the airframe count as attached (brackets, mounts), so a part can float up to that far away.
- **Design environment speed:** a step costs about 8 to 25 s (a new aerodynamic shape costs about 6 s extra, cached on disk afterwards), so training an RL
  policy on it needs `fidelity="analytic"` for pre-training or patience; black-box optimisers and LLM agents are the realistic users.

## Regenerate

```
.venv/bin/python -m pytest -q
.venv/bin/python validation/falcon_claim.py
.venv/bin/dronebench score fixtures/falcon_v2 --flown --summary
.venv/bin/python -m dronebench.train --task orbit --steps 800000 --out runs/ppo_orbit
```
