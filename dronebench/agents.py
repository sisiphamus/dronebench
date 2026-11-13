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


def run_claude(ws, steps: int = 12, model: Optional[str] = None, say: Callable[[str], None] = print,
               max_turns: int = 80) -> int:
    ws = Path(ws).resolve()
    claude = shutil.which("claude")
    if not claude:
        raise W.WorkspaceError("`claude` (Claude Code) is not on PATH")
    bindir = str(Path(sys.executable).resolve().parent)
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""), DRONEBENCH_AGENT="claude")
    cmd = [claude, "-p", claude_prompt(ws, steps), "--max-turns", str(max_turns),
           "--allowedTools", "Bash(dronebench:*)", "Bash(ls:*)", "Bash(cat:*)", "Read", "Write", "Glob", "Grep"]
    if model:
        cmd += ["--model", model]
    say(f"launching Claude Code in {ws} ({steps} steps)")
    p = subprocess.run(cmd, cwd=ws, env=env)
    return p.returncode
