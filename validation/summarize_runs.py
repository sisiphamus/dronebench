"""Summarise agent runs (workspaces) into one JSON file for the benchmark table on the site.

    .venv/bin/python validation/summarize_runs.py runs/claude-falcon runs/cma-falcon ... --out docs/data/leaderboard.json
"""
import argparse
import json
from pathlib import Path

LABELS = {"claude": "Claude Code", "cma": "CMA-ES (design parameters)", "random-params": "Random search (design parameters)",
          "random": "Random part edits"}


def summarize(ws: Path) -> dict:
    meta = json.loads((ws / "workspace.json").read_text())
    recs = [json.loads(l) for l in (ws / "runs.jsonl").read_text().splitlines() if l.strip()]
    steps = [r for r in recs if r.get("step", 0) > 0]
    agent = next((r.get("agent") for r in steps if r.get("agent")), "agent")
    best, curve = recs[0]["score"], []
    for r in recs:
        if r.get("accepted"):
            best = max(best, r["score"])
        curve.append(round(best, 3))
    return {"run": ws.name, "agent": agent, "label": LABELS.get(agent, agent), "fixture": meta["fixture"], "preset": meta["preset"],
            "baseline": round(recs[0]["score"], 2), "best": round(best, 2), "gain_pct": round((best / recs[0]["score"] - 1) * 100, 1),
            "attempts": len(steps), "accepted": sum(1 for r in steps if r.get("accepted")),
            "gate_failures": sum(1 for r in steps if r.get("reason") == "gate failed"),
            "worse": sum(1 for r in steps if (r.get("reason") or "").startswith("regression")),
            "invalid": sum(1 for r in steps if r.get("reason") and r["reason"].startswith("invalid")),
            "best_so_far": curve}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [summarize(Path(r)) for r in a.runs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['label']:<36} {r['baseline']:>7.1f} -> {r['best']:>7.1f} ({r['gain_pct']:+.0f}%)  {r['attempts']} attempts, "
              f"{r['gate_failures']} failed the gate, {r['worse']} scored lower")
