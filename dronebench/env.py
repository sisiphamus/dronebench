"""Gymnasium-style environment over design folders.

    env = DroneBench("fixtures/falcon_v2", workdir="runs/falcon")
    obs = env.reset()                       # the report dict (removal test on for this first obs)
    obs, reward, done, info = env.step([{"op": "remove", "id": "ballast_nose"}])

Rejection policy: the agent always stays on its last good design.
  * invalid ops (apply_edits errors)   -> rejected, reward 0, info["errors"]
  * the edited design fails the gate   -> rejected, reward 0, info["failures"] (check name + detail),
                                          info["score"] = 0 and info["candidate_report"] = its report
  * otherwise                          -> accepted, reward = new score - old score, the design advances.
A step never makes the current design worse at the gate. Every attempted step counts toward max_steps.
Each accepted or rejected candidate lives in workdir/step_N so it can be inspected.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional


class DroneBench:
    def __init__(self, design_dir, preset: Optional[str] = None, workdir=None, necessity: bool = False,
                 max_steps: int = 50, runs_path=None, agent: str = "env"):
        self.origin = Path(design_dir)
        self.preset = preset
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="dronebench_"))
        self.necessity = necessity
        self.max_steps = max_steps
        self.runs_path = runs_path
        self.agent = agent
        self.current: Path = self.origin
        self.report: Optional[dict] = None
        self.steps = 0

    # ------------------------------------------------------------ helpers
    def _evaluate(self, d: Path, necessity: bool) -> dict:
        from .pipeline import evaluate
        return evaluate(d, preset=self.preset, necessity=necessity)

    @staticmethod
    def score_of(report: Optional[dict]) -> float:
        v = ((report or {}).get("score") or {}).get("value")
        return float(v) if isinstance(v, (int, float)) else 0.0

    def _log(self, report: Optional[dict], ops, design: Path, accepted: bool, errors=None) -> None:
        if not self.runs_path:
            return
        from .runs import append_run, record_from_report
        if report is not None:
            rec = record_from_report(report, step=self.steps, ops=ops, agent=self.agent, design=str(design))
        else:
            rec = {"step": self.steps, "design": str(design), "preset": self.preset, "score": 0.0, "gate_pass": False,
                   "failures": ["invalid_ops"], "ops": ops, "agent": self.agent}
        rec["accepted"] = accepted
        if errors:
            rec["errors"] = errors
        append_run(self.runs_path, rec)

    # ------------------------------------------------------------ API
    def reset(self) -> dict:
        self.current = self.origin
        self.steps = 0
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.report = self._evaluate(self.current, necessity=True)
        if self.preset is None:
            self.preset = self.report.get("preset")
        self._log(self.report, [], self.current, True)
        return self.report

    def step(self, ops: list) -> tuple[dict, float, bool, dict]:
        from .edits import apply_edits
        if self.report is None:
            self.reset()
        self.steps += 1
        done = self.steps >= self.max_steps
        old = self.score_of(self.report)
        out = self.workdir / f"step_{self.steps}"
        info: dict = {"step": self.steps, "path": str(out), "rejected": False, "errors": [], "failures": [],
                      "score": old, "previous_score": old}
        new_dir, errors = apply_edits(self.current, ops, out)
        if errors:
            info.update(rejected=True, errors=errors, path=str(self.current), reason="invalid ops")
            self._log(None, ops, out, False, errors)
            return self.report, 0.0, done, info
        try:
            rep = self._evaluate(new_dir, necessity=self.necessity)
        except Exception as e:
            info.update(rejected=True, errors=[f"evaluation failed: {type(e).__name__}: {e}"],
                        path=str(self.current), reason="evaluation failed")
            self._log(None, ops, out, False, info["errors"])
            return self.report, 0.0, done, info
        new = self.score_of(rep)
        fails = [c for c in rep["gate"]["checks"] if not c.get("pass")]
        info["score"] = new
        info["candidate_path"] = str(new_dir)
        if fails:
            info.update(rejected=True, reason="gate failed", path=str(self.current), candidate_report=rep,
                        failures=[f"{c['name']}: {c.get('detail') or ''}".rstrip(": ") for c in fails])
            self._log(rep, ops, new_dir, False)
            return self.report, 0.0, done, info
        self.current, self.report = new_dir, rep
        info["path"] = str(new_dir)
        self._log(rep, ops, new_dir, True)
        return rep, new - old, done, info
