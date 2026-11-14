"""Fit the one explicit tuning knob to measurements from the real aircraft.

DroneBench takes its drag from AeroSandbox and adds a single factor, `cd0_factor`, for what a CAD-based model cannot
see: seams, gaps, control-surface hinges, print roughness. It is 1.0 by default. If you have a few measurements of
battery power in steady level flight (from a flight log or a bench test), this finds the factor that matches them:

    dronebench calibrate fixtures/falcon_v2 --point 12.5:70 --point 15.3:92 --point 18:124      # airspeed m/s : battery W

Use the printed factor with `Vehicle.from_bits(..., cd0_factor=f)` or set `cd0_factor` in the workspace.
The fit is one number; if the residuals are large the model structure (not the factor) is what is wrong.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_cd0_factor(design_dir, points: Sequence[tuple], lo: float = 0.4, hi: float = 3.0) -> dict:
    """points: [(airspeed m/s, measured battery watts), ...]. Returns {"factor", "rms_error_pct", "rows"}."""
    from scipy.optimize import minimize_scalar

    from . import flight as Fl
    from . import mission as M
    from .ingest import load_design
    from .pipeline import apply_plausibility, core

    d = load_design(design_dir)
    apply_plausibility(d)
    bits = core(d, d.mission.preset or "endurance")

    def predict(f):
        veh = Fl.Vehicle.from_bits(bits, d, cd0_factor=f)
        out = []
        for V, _ in points:
            r = M.power_at(veh, float(V), soc=0.8)
            out.append(float("nan") if r is None else r["P"])
        return np.array(out)

    meas = np.array([w for _, w in points], float)

    def loss(f):
        p = predict(f)
        return float(np.nanmean(((p - meas) / meas) ** 2))

    res = minimize_scalar(loss, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    f = float(res.x)
    p = predict(f)
    rows = [{"V": float(V), "measured_w": float(w), "model_w": float(pp), "error_pct": float((pp - w) / w * 100)}
            for (V, w), pp in zip(points, p)]
    return {"factor": f, "rms_error_pct": float(np.sqrt(np.nanmean([(r["error_pct"]) ** 2 for r in rows]))), "rows": rows,
            "uncalibrated_rms_error_pct": float(np.sqrt(np.nanmean(((predict(1.0) - meas) / meas * 100) ** 2)))}
