# DroneBench

A benchmark environment where an AI agent designs a small electric fixed-wing drone, tests it, and iterates.

DroneBench reads a design (CAD meshes plus a parts list), checks that it can fly, scores it, and explains every part: what it does, why it is there, whether the aircraft still flies without it. It never suggests changes. The agent edits the design through a command, DroneBench accepts or rejects the edit with reasons, and a live dashboard shows the whole run.

```
agent (Claude Code, your script, a random baseline)
   │  dronebench try / build            ← edit the design
   ▼
analysis: ingest → part graph → geometry → mass → aero + propulsion + structure → 15-check gate → score
   │  accept (gate passes, score not worse)  or  reject (check name, value, limit)
   ▼
workspace on disk  ──►  live dashboard (3D model, score curve, step log, gate headroom)
```

## Quick start

```bash
uv venv && uv pip install -e .

dronebench init runs/falcon --fixture falcon_v2 --steps 12     # workspace: design, TASK.md, step log
dronebench serve runs/falcon --open                            # live dashboard on http://127.0.0.1:8765

# in another terminal, let an agent loose:
dronebench run runs/falcon --agent claude --steps 12           # Claude Code, headless
dronebench run runs/falcon --agent random --steps 12           # random-search baseline
```

Or drive it by hand (this is exactly what an agent does):

```bash
cd runs/falcon
dronebench status .                                            # score, tightest checks, removable parts
echo '[{"op":"remove","id":"ballast_nose"}]' > ops.json
dronebench try . --ops ops.json --note "ballast is dead weight"    # edit → score → accept/reject
dronebench build . --set wing.span_mm=2200 --note "more span"      # change a design parameter, rebuild the aircraft
```

## The loop

Each attempt is one step. The rule is the same for every agent:

| Outcome | When | Effect |
|---|---|---|
| accepted | gate passes and score ≥ best so far | `design/` becomes this attempt |
| rejected: gate failed | any of the 15 checks fails | nothing changes; the reply names the check, value and limit |
| rejected: regression | gate passes but score dropped | nothing changes (`--keep` overrides, for setup steps) |
| rejected: invalid | bad ops, unknown part, failed build | nothing changes; the reply says why |

Every attempt, accepted or not, is kept in `steps/N/` (design, report, ops), so any step can be inspected, replayed or diffed. Accepted steps get the full removal test (each part is removed in a copy and the aircraft re-analysed), so `dronebench status` always lists what can still be cut.

## What the agent can change

Two ways, both through the CLI:

- **`try`**: JSON edit ops on the parts list and meshes: `remove`, `swap` (new part number and specs), `set` (any field), `scale`, `move`, `add`. Specs an agent types for a new part are tagged `agent_supplied` and checked for plausibility (battery Wh/kg, motor power per gram, …); implausible values are clamped, so inventing a 400 Wh/kg battery does not help.
- **`build`**: for parametric designs (`falcon_v2`), change `params.yaml` knobs (span, chords, airfoils, tail size and arm, spar tubes, motor, prop, battery, …) and the whole aircraft, CAD included, is regenerated.

Locked parts (the mission payload) and `mission.yaml` cannot be changed.

## Fixtures

| Name | What | Parts |
|---|---|---|
| `falcon_v2` | Titan Falcon V2 replica, parametric, with seeded waste (ballast, spare GPS, armor, doubler) | 80 |
| `trainer` | small trainer | 42 |
| `bad_tail` | trainer with an undersized tail (fails the gate; a repair task) | 42 |
| `step_mix` | trainer with STEP parts mixed in | 42 |
| `third_party` | real downloaded CAD (`fixtures/fetch_third_party.py`, needs network) | 48 |

Presets (what gets maximised): `endurance` (minutes), `range` (km), `speed` (km/h), `payload` (payload fraction), `efficiency` (km/Wh). `dronebench init --preset range` changes it.

## Dashboard

`dronebench serve` is read-only and polls the workspace files, so it works while an agent is running in another process.

- **3D model**, coloured by necessity (required / helpful / removable), by function, or by what changed against the baseline (changed, added, removed shown as a red ghost). Baseline ghost, explode slider, click any part for its report.
- **Score curve** with the best-so-far line; rejected steps marked (× gate failed, ○ regression). Click a point to jump to that step.
- **Attempts** table: result, delta, the agent's one-line reason, and why a rejection happened.
- **Gate** panel: every check with its value, limit and headroom bar.
- **Replay** walks through the accepted designs; ←/→ step through attempts.

## Python API

```python
from dronebench import workspace as W
W.init("runs/x", "falcon_v2", steps=12)
out = W.try_step("runs/x", [{"op": "remove", "id": "ballast_nose"}], note="dead weight")
out["accepted"], out["delta"], out["failures"]
```

There is also the Gymnasium-style `dronebench.env.DroneBench` (`reset()` / `step(ops)`, reward = score change) for training or evaluation harnesses, and the one-shot commands `dronebench score|parts|view|edit|baseline` for scoring a single design folder.

## How much to trust it

All physics is analytic (NACA lift slopes, drag build-up, motor + prop + battery model, cantilever spar, neutral-point stability). It runs in about 2 seconds per design and is meant for ranking designs against each other, with an error band of about ±15%. The Falcon V2 replica at 3.5 kg lands at 1.7 Wh/km against a published 1.6. It is not a certificate of airworthiness, and the replica's planform is an assumption (see `fixtures/README.md`).

## Layout

```
dronebench/   ingest, graph, geometry, physics/, gate, necessity, report   (the benchmark)
              workspace.py agents.py server.py viewer/                    (the agent loop and dashboard)
fixtures/     design folders and the scripts that generate them
tests/        pytest suite (`pytest -q`)
PLAN.md       original build plan and data contracts
```
