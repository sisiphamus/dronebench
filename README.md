# DroneBench

An environment for small electric fixed-wing drones where every design is **flown** in a simulator before it scores. It has two halves that
share one simulator:

- **Design environment.** An agent (an LLM, a black-box optimiser, an RL policy) edits an aircraft. DroneBench builds the
  aerodynamic model from the geometry, flies the design in a 6-degree-of-freedom simulator with wind and turbulence, and keeps
  or rejects the edit with the reason. Reward is the change in a mission score that was measured in flight.
- **Flight environment.** A Gymnasium environment for learning to fly a given design: continuous control of aileron, elevator,
  rudder and throttle at 50 Hz, 3,000 to 5,000 steps per second per core (the compiled simulator alone does 60,000).

```
edit  ->  geometry  ->  AeroSandbox aero model  ->  6-DOF simulator  ->  autopilot flies climb, orbits, legs
 ^                      (vortex lattice, NeuralFoil)   (numba RK4)          in calm, light and strong wind + turbulence
 |                                                                                  |
 +---- kept or rejected, with the check, value and limit  <-- gate  <-- measured endurance, range, climb, loads
```

What it will and will not tell you is in [VALIDATION.md](VALIDATION.md). Short version: the physics is checked against theory,
against two independent aerodynamic methods, and against one manufacturer figure (without tuning). It has not been compared with
measured flight logs, and it only covers conventional wing + tail layouts.

## Quick start

```bash
uv venv && uv pip install -e .                                   # adds AeroSandbox, numba, gymnasium
dronebench init runs/demo --fixture falcon_v2 --steps 12         # a workspace: design, brief, step log
dronebench serve runs/demo --open                                # live dashboard
dronebench run runs/demo --agent claude                          # in another terminal: Claude Code iterates
```

Other agents on the same workspace: `--agent cma` (CMA-ES over the design parameters), `--agent random-params`, `--agent random` (random part edits).
Or drive it by hand, which is what an agent does:

```bash
dronebench status runs/demo
dronebench try   runs/demo --ops ops.json --note "ballast is dead weight"     # edit parts, fly, accept or reject
dronebench build runs/demo --set wing.span_mm=2300 --set htail.area_m2=0.09  # change parameters, rebuild, fly
dronebench score fixtures/falcon_v2 --flown --summary                          # fly one design folder
```

## As a Gymnasium environment

```python
import gymnasium as gym, dronebench.envs

# learn to design: each step edits parameters, rebuilds, flies (about 20 s), reward = change in score
env = gym.make("DroneBench/Design-v0", fixture="falcon_v2", max_steps=20)

# learn to fly: continuous control in wind and turbulence
env = gym.make("DroneBench/Flight-v0", design="fixtures/falcon_v2", task="orbit")   # or "line", "hold"
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.unwrapped.expert_action()            # the design-aware cascaded autopilot, as a reference policy
```

`python -m dronebench.train --task orbit` trains PPO (stable-baselines3, optional) and writes a learning curve with the autopilot,
zero-action and random policies as reference lines.

## What gets flown, and what is measured

| Segment | Weather | Used for |
|---|---|---|
| Full-throttle climb at best-climb speed | calm | rate of climb (score and gate) |
| Orbit, 150 m radius, 130 s, at 100% and 50% charge | calm, and a steady 5 m/s wind | endurance: usable energy / mean battery power |
| Upwind and downwind legs | calm, and a steady 5 m/s wind | range and Wh/km: mean of the two legs' energy per km |
| Orbit and legs in turbulence | light (3 m/s wind, Dryden sigma 1) and gusty (6 m/s, sigma 2) | **pass/fail only**: completes, tracks altitude and speed, stays under the load and stall limits |

The scored segments are deterministic (no random turbulence), so a design's score is identical every run. Random turbulence is flown for
pass or fail, not scored: one realisation moves average power by tens of percent. Same weather for every design, so scores compare;
`seed_offset` rotates the turbulence seeds if you want to hold some back.

The gate checks static margin, stall margin, climb, that every segment is completed, peak load factor, no stall, a pack that fits its volume, the
mission's takeoff mass limit, and structure and systems: spar, current limits, wiring, every part attached, continuous fuselage, printed shells with ribs.
An aircraft that cannot be trimmed, cannot be flown by the autopilot, or has no tail the model can represent fails with the reason, instead of
receiving a made-up score.

## Baselines on the Falcon V2 (endurance, minutes)

One run each from the same starting design, same weather and gate; scores are flown. Reproduce with `dronebench run WS --agent claude|cma|random-params|random`.

| Agent | Start | Best | Attempts | Failed the gate | Scored lower |
|---|---|---|---|---|---|
| Claude Code | 134 | 204 (+53%) | 15 | 4 | 1 |
| Random part edits | 134 | 146 (+9%) | 30 | 9 | 7 |
| Random search over design parameters | 134 | 143 (+7%) | 30 | 22 | 4 |
| CMA-ES over design parameters | 134 | 142 (+7%) | 30 | 12 | 14 |

A learned flight controller: PPO on `FlightEnv` (orbit in wind and turbulence) reaches a return of 138 against the cascaded autopilot's 152 after 800,000
steps (about 30 minutes on one core), completing all 8 held-out test flights; doing nothing returns -894 and random actions -207.
One run per agent gives a sanity check that the environment rewards engineering over luck, with no ranking between the search methods.

## What an agent can change

- **`try`**: JSON ops on the parts list and meshes: `remove`, `swap` (new part number and specs), `set`, `scale`, `move`, `add`. Specs an
  agent types for a new part are tagged `agent_supplied` and checked for plausibility (battery Wh/kg, power per gram); implausible values are clamped.
- **`build`**: for parametric designs (`falcon_v2`) the builder's knobs: span, chords, taper break, dihedral, tail area and arm, fin area, shell
  thickness, motor, prop and battery choice. The whole aircraft, CAD included, is regenerated.

Locked parts (the mission payload) and `mission.yaml` cannot be changed.

## Calibrating to a real aircraft

DroneBench has one explicit tuning knob, a drag factor for what CAD cannot see (seams, hinge gaps, print roughness). With a few
steady-flight measurements of battery power:

```bash
dronebench calibrate fixtures/falcon_v2 --point 15.3:92 --point 18:124      # airspeed m/s : battery watts (use your own log)
```

## Fixtures

| Name | What | Parts |
|---|---|---|
| `falcon_v2` | Titan Falcon V2 replica, parametric, with seeded dead weight | 80 |
| `trainer` | small trainer | 42 |
| `bad_tail` | trainer with an undersized tail (statically unstable; a repair task) | 42 |
| `step_mix` | trainer with STEP parts mixed in | 42 |
| `third_party` | real downloaded CAD, no tail (flying wing); fails with a message | 48 |

## Dashboard

`dronebench serve` is read-only and polls the workspace, so it works while an agent runs in another process: the 3D aircraft (needed, helps,
removable; what changed against the baseline), the score curve with rejected steps marked, every attempt with the reason, and every gate check
with its headroom, including the flown ones. `dronebench export WS --out site/demo` writes a static copy for hosting.

## Layout

```
dronebench/aero.py          AeroSandbox model from geometry, coefficient tables (cached by geometry hash)
dronebench/flight.py        6-DOF simulator, powertrain, trim; _dyn.py is the compiled inner loop
dronebench/autopilot.py     cascaded autopilot with gains from the design's own derivatives
dronebench/mission.py       flown climb / orbit / legs, performance sweep, flight checks
dronebench/flown.py         merges the flown evaluation into the report
dronebench/envs.py          Gymnasium environments
dronebench/workspace.py     the agent loop: init, try, build, status
dronebench/server.py viewer/   live dashboard
dronebench/ (the rest)      ingest, part graph, geometry, closed-form physics, structure and systems gate, removal test
fixtures/  validation/  tests/
VALIDATION.md               what was checked and what is open
```
