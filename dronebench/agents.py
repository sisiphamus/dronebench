"""Built-in agents that drive a workspace, so the loop can run end to end without a human.

  random  - hill climb over random edits (the baseline an LLM has to beat)
  claude  - Claude Code in headless mode, given TASK.md and the `dronebench` commands

Anything else (another CLI agent, your own script) can drive a workspace the same way: read
TASK.md, call `dronebench try/build`, read the outcome. Or use `workspace.try_step` from Python.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import workspace as W


def run_random(ws, steps: int = 12, seed: int = 0, say: Callable[[str], None] = print) -> dict:
    from .baseline import propose
    ws = Path(ws).resolve()
    rng = random.Random(seed)
    m = W.meta(ws)
    start = m["best"]
    left = min(steps, m["max_steps"] - m["steps"])
    for _ in range(left):
        rep = W.current_report(ws)
        ops = None
        for _ in range(10):
            ops = propose(rep, rng)
            if ops:
                break
        if not ops:
            break
        o = W.try_step(ws, ops, note=_describe(ops), agent="random")
        say(W.outcome_text(o))
    return {"start": start, "best": W.meta(ws)["best"]}


def _param_search(ws, steps: int, seed: int, strategy: str, say) -> dict:
    """Black-box search over the builder's parameters; every candidate is a workspace step (kept or rejected as usual)."""
    import yaml
    from .envs import ParamCoder
    ws = Path(ws).resolve()
    m = W.meta(ws)
    if not m.get("has_params"):
        raise W.WorkspaceError("this design is not parametric; param search needs a params.yaml")
    coder = ParamCoder()
    x0 = coder.encode(yaml.safe_load((ws / "params.yaml").read_text())).astype(float)
    rng = np.random.default_rng(seed)
    start = m["best"]
    left = min(steps, m["max_steps"] - m["steps"])
    if strategy == "cma":
        import cma
        es = cma.CMAEvolutionStrategy(x0, 0.18, {"seed": seed + 1, "bounds": [-1, 1], "verbose": -9, "popsize": 6})
    done = 0
    while done < left:
        xs = es.ask() if strategy == "cma" else [np.clip(x0 + rng.normal(0, 0.3, x0.shape), -1, 1) for _ in range(left - done)]
        fs = []
        for x in xs:
            if done >= left:
                fs.append(0.0)
                continue
            o = W.build_step(ws, coder.decode(x), note=f"{strategy}", agent=strategy)
            say(W.outcome_text(o))
            done += 1
            fs.append(-(o["score"] or 0.0))
        if strategy == "cma":
            es.tell(xs, fs)
        else:
            break
    return {"start": start, "best": W.meta(ws)["best"]}


def run_cma(ws, steps: int = 12, seed: int = 0, say: Callable[[str], None] = print) -> dict:
    return _param_search(ws, steps, seed, "cma", say)


def run_random_params(ws, steps: int = 12, seed: int = 0, say: Callable[[str], None] = print) -> dict:
    return _param_search(ws, steps, seed, "random-params", say)


def _describe(ops: list) -> str:
    bits = []
    for o in ops[:3]:
        k = o.get("op")
        t = o.get("id") or ",".join(o.get("ids", [])[:2])
        bits.append(f"{k} {t}")
    return "random: " + "; ".join(bits)


def claude_prompt(ws: Path, steps: int) -> str:
    return (f"Read TASK.md in this folder and do the task. You have {steps} dronebench steps. "
            "Work only through the `dronebench` commands (status, parts, try, build). Write ops files with the "
            "Write tool. After every attempt, read the outcome before choosing the next change. "
            "End with the summary TASK.md asks for.")


def run_command(ws, command: str, label: str = "agent", say: Callable[[str], None] = print) -> int:
    """Run any command as the agent, inside the workspace. This is how another agent is plugged in.

    The command gets the workspace as its working directory, `dronebench` on PATH, and:
      DRONEBENCH_WORKSPACE  the workspace folder      DRONEBENCH_AGENT  the label written to the step log
      DRONEBENCH_STEPS      the step budget           (TASK.md in the folder is the brief)
    Its exit code is returned. The step budget is enforced by the workspace, not by the command.
    """
    ws = Path(ws).resolve()
    m = W.meta(ws)
    bindir = str(Path(sys.executable).parent)  # not resolved: a venv python is a symlink to the real one
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""), DRONEBENCH_AGENT=label,
               DRONEBENCH_WORKSPACE=str(ws), DRONEBENCH_STEPS=str(m["max_steps"]))
    say(f"running `{command}` as '{label}' in {ws} ({m['max_steps']} steps)")
    return subprocess.run(command, cwd=ws, env=env, shell=True).returncode


def run_claude(ws, steps: int = 12, model: Optional[str] = None, say: Callable[[str], None] = print,
               max_turns: int = 80) -> int:
    ws = Path(ws).resolve()
    claude = shutil.which("claude")
    if not claude:
        raise W.WorkspaceError("`claude` (Claude Code) is not on PATH")
    bindir = str(Path(sys.executable).parent)  # not resolved: a venv python is a symlink to the real one
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""), DRONEBENCH_AGENT="claude")
    cmd = [claude, "-p", claude_prompt(ws, steps), "--max-turns", str(max_turns),
           "--allowedTools", "Bash(dronebench:*)", "Bash(ls:*)", "Bash(cat:*)", "Read", "Write", "Glob", "Grep"]
    if model:
        cmd += ["--model", model]
    say(f"launching Claude Code in {ws} ({steps} steps)")
    p = subprocess.run(cmd, cwd=ws, env=env)
    return p.returncode
