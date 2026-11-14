# DroneBench

A test bench for AI agents that design small electric drones. An agent changes an aircraft, DroneBench flies the result in a
6-degree-of-freedom simulator with wind and turbulence, and says whether the change was kept and why. The score is what the aircraft
did in flight (minutes of endurance by default), and a design that fails a safety check scores nothing. You bring the agent; the
bench supplies the aircraft, the physics, the rules and a record of every attempt.

Two halves share one simulator:

- **Design environment.** An agent (an LLM, a black-box optimiser, an RL policy) edits an aircraft. DroneBench builds the
  aerodynamic model from the geometry, flies it, and keeps or rejects the edit with the reason. Reward is the change in a mission score
  that was measured in flight.
- **Flight environment.** A Gymnasium environment for learning to fly a given design: continuous control of aileron, elevator,
  rudder and throttle at 50 Hz.

```
edit  ->  geometry  ->  AeroSandbox aero model  ->  6-DOF simulator  ->  autopilot flies climb, orbits, legs
 ^                      (vortex lattice, NeuralFoil)   (numba RK4)          in calm, light and strong wind + turbulence
 |                                                                                  |
 +---- kept or rejected, with the check, value and limit  <-- gate  <-- measured endurance, range, climb, loads
```

What it will and will not tell you is in [VALIDATION.md](VALIDATION.md). Short version: the physics is checked against theory,
against two independent aerodynamic methods, and against one manufacturer figure (without tuning). It has not been compared with
measured flight logs, and it only covers conventional wing + tail layouts.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/sisiphamus/dronebench && cd dronebench
uv venv && uv pip install -e .          # or: python -m venv .venv && .venv/bin/pip install -e .
source .venv/bin/activate
dronebench score fixtures/trainer --flown --summary      # flies one design and prints the score and gate (a minute the first time, while aerodynamic tables are computed and cached)
```

## Test your agent

A run happens in a **workspace**: a folder holding the current design, a brief for the agent (`TASK.md`), and a log of every attempt.
The agent never edits files itself; it calls `dronebench try` or `dronebench build`, and the workspace keeps the change only if the
design passes the gate and scores at least as well (or the agent adds `--keep` for a setup step that should stay even if the score dips). The step budget is enforced by the workspace.

```bash
dronebench init ws --fixture falcon_v2 --steps 30          # the standard task: improve endurance of the Falcon V2
dronebench serve ws --open                                 # live dashboard in a browser (optional)
dronebench run ws --cmd "python my_agent.py" --label my-agent
```

`--cmd` takes any command. It is started inside the workspace with `DRONEBENCH_WORKSPACE`, `DRONEBENCH_STEPS` and `DRONEBENCH_AGENT`
set, and `dronebench` on its PATH. A coding agent reads `TASK.md` (it explains the commands and the goal); a script can call the
commands directly or use the Python functions:

```python
from dronebench import workspace as W

o = W.try_step("ws", [{"op": "remove", "id": "wiring_signal"}], note="why", agent="my-agent")   # edit parts
o = W.build_step("ws", {"wing.span_mm": 2200}, note="why", agent="my-agent")                   # change design numbers
o["accepted"], o["score"], o["delta"], o["failures"]        # kept?, flown score, change, which checks failed and by how much
W.status("ws")                                             # score, tightest gate margins, parts the removal test calls unnecessary
```

Built in: `--agent claude` (Claude Code), `cma` (CMA-ES over the design numbers), `random-params`, `random` (random part edits).

Start from your own numbers rather than the fixture's, or from your own design folder ([DESIGN_FORMAT.md](DESIGN_FORMAT.md)):

```bash
dronebench init ws --fixture falcon_v2 --set wing.span_mm=1900 --set htail.area_m2=0.06
dronebench init ws --fixture path/to/my_design
```

### Score it and share it

```bash
dronebench result ws --out results/falcon_v2/my-agent.json    # the agent, every step, every score
dronebench verify results/falcon_v2/my-agent.json             # re-flies the run from the start and compares every score
```

`verify` does not trust the file: it rebuilds the starting design, replays each accepted step through the same gate and simulator, and
fails if any score is off by more than 0.1%. `--standard` also checks that it is the standard task (plain `falcon_v2`, default mission, at most 30 steps). Results are submitted as pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md).

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
mission's takeoff mass limit, and structure and systems: spar, current limits, wiring, flight controller, receiver and GPS, a hinge and a linkage on every control surface, every part attached, continuous fuselage, printed shells with ribs.
An aircraft that cannot be trimmed, cannot be flown by the autopilot, or has no tail the model can represent fails with the reason, instead of
receiving a made-up score.

## Baselines on the Falcon V2 (endurance, minutes)

One run each of 30 steps from the same plain `falcon_v2`, same weather and gate; scores are flown. The files are in `results/falcon_v2/` and replay
with `dronebench verify --standard`. Reproduce with `dronebench run WS --agent claude|cma|random-params|random`.

| Agent | Start | Best | Attempts | Failed the gate | Scored lower |
|---|---|---|---|---|---|
| Claude Code | 149 | 345 (+132%) | 30 | 8 | 0 |
| Random part edits | 149 | 168 (+13%) | 30 | 10 | 7 |
| Random search over design parameters | 149 | 161 (+8%) | 30 | 22 | 5 |
| CMA-ES over design parameters | 149 | 159 (+7%) | 30 | 12 | 14 |

Claude Code's 345 minutes came from a 3.5 times larger battery pack carried by a longer wing (2.09 m to 2.8 m), a stronger spar, smaller propellers
to stay under the motor current limit, the pack moved aft to cut trim drag, and lighter printed shells, all inside the 7 kg takeoff limit and with every
gate check passing. The gate now requires a GPS, a hinge and a linkage on every control surface and a minimum of servo wiring, so removing parts the
simulator does not need no longer scores; an airspeed sensor and control horns can still be dropped for a minute or two.

A learned flight controller: PPO on `FlightEnv` (orbit in wind and turbulence) trained on the 3.4 kg `falcon_v2_padded` reaches a return of 138 against the
cascaded autopilot's 152 after 800,000 steps (about 30 minutes on one core), completing all 8 held-out test flights; doing nothing returns -894 and random
actions -207. One run per agent gives a sanity check that the environment rewards engineering over luck, with no ranking between the search methods.

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
| `falcon_v2` | Titan Falcon V2 replica, parametric. The standard task | 74 |
| `falcon_v2_padded` | the same with planted dead weight (ballast, second GPS, armor, fairings): a clean-up task | 80 |
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
