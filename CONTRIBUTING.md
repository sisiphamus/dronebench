# Contributing

## Run the tests

```bash
git clone https://github.com/sisiphamus/dronebench && cd dronebench
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # about two minutes
```

## Submit a result for your agent

1. Run your agent on the standard task, a fresh workspace with 30 steps:
   ```bash
   dronebench init ws --fixture falcon_v2 --steps 30
   dronebench run ws --cmd "python my_agent.py" --label my-agent
   ```
2. Export it: `dronebench result ws --out results/falcon_v2/my-agent.json`
3. Check that it replays on your machine: `dronebench verify --standard results/falcon_v2/my-agent.json`
4. Open a pull request with the file. CI replays every result file in the pull request and fails if a score does not reproduce.

Rules for a result to count, all checked by `verify --standard`: the unmodified `falcon_v2` with the default mission, flown, at most 30 steps,
and no `add` op that reads a mesh from your disk (`cad_file`), because nobody else has that file. Use only `dronebench try` and
`dronebench build`; edits made outside them do not replay and fail the check. Say in the pull request which model or method it was.
One run is a sample, not a ranking; repeat runs are welcome. Scores must reproduce to 0.1%. Replay can prove the accepted steps but not the rejected ones, so the attempt count rests on the submitter being honest: steps must be numbered 1 to N with none missing, and say in the pull request how many attempts you ran.

## Add a design or a fixture

Follow [DESIGN_FORMAT.md](DESIGN_FORMAT.md). A fixture needs meshes that are watertight, a parts list the gate passes, and a test in
`tests/test_fixtures.py`. Do not plant parts that exist only to be removed in a default fixture; put them in a separate `_padded` variant.

## Change the physics

Say what the change is checked against (theory, a second method, a measurement) in the pull request and add the check to `VALIDATION.md`.
Scores are deterministic, so a physics change moves every number on the leaderboard; old results are replayed and refreshed in the same
pull request.
