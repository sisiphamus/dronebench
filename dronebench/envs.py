"""Gymnasium environments.

    FlightEnv   learn to fly a given design: continuous control of aileron, elevator, rudder and throttle in wind and
                turbulence, holding an orbit, a line or an altitude. 50 Hz control, about 3 000 to 5 000 steps per second
                per core (the compiled simulator alone does 60 000).
    DesignEnv   learn to design: each step edits the aircraft (parameters of the parametric builder), the design is
                rebuilt and flown, the reward is the change in mission score. The agent loop of `dronebench try/build`
                as a Gymnasium environment.

Both register with gymnasium:  gym.make("DroneBench/Flight-v0", design="fixtures/falcon_v2"),
gym.make("DroneBench/Design-v0", fixture="falcon_v2").
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import autopilot as AP
from . import flight as Fl
from . import mission as M

CTRL_DT = 0.02
SUBSTEPS = 2


def _vehicle(design) -> Fl.Vehicle:
    """Vehicle from a design folder, a Design, or an existing Vehicle."""
    if isinstance(design, Fl.Vehicle):
        return design
    from .ingest import load_design
    from .pipeline import apply_plausibility, core
    d = load_design(Path(design)) if not hasattr(design, "parts") else design
    apply_plausibility(d)
    return Fl.Vehicle.from_bits(core(d, d.mission.preset or "endurance"), d)


class FlightEnv(gym.Env):
    """Low-level flight control.

    tasks: "orbit"      circle of radius `radius` m at `alt` m and the design airspeed
           "line"       straight track at `alt`, crosswind
           "hold"       hold altitude and airspeed, wings level, no path

    Observation (float32, 20): altitude error/20, airspeed error/5, sin/cos of roll, pitch and of the course error, body
    rates p q r (rad/s)/2, alpha, beta, cross-track error/50, course error, the three surface positions, state of charge,
    load factor - 1.
    Action (4, in [-1, 1]): aileron, elevator, rudder around trim (full range = 25 deg), throttle around trim (+-0.5).
    Reward per control step: 0.1 minus 0.1 x (altitude error/10 + airspeed error/3 + track error/40), each term capped at 3,
    so a perfect flight earns 0.1 per step and a drifting one earns less or goes negative. -100 and the episode ends on a
    crash, a structural failure, a stall, or leaving the 80 m altitude corridor (leaving is not a way to stop paying penalties).
    """
    metadata = {"render_modes": []}

    def __init__(self, design="fixtures/falcon_v2", task: str = "orbit", radius: float = 150.0, alt: float = 100.0,
                 episode_s: float = 60.0, wind_max: float = 6.0, turb_levels=(0.0, 1.0, 2.0), randomize: bool = True,
                 airspeed: Optional[float] = None):
        super().__init__()
        self.veh = _vehicle(design)
        self.task, self.radius, self.alt0 = task, radius, alt
        self.episode_s, self.wind_max, self.turb_levels, self.randomize = episode_s, wind_max, turb_levels, randomize
        self.V = float(airspeed or max(self.veh.v_design, 1.45 * M.stall_speed(self.veh)))
        self.ap = AP.Autopilot(self.veh, self.V, alt=alt)           # trim values and the expert controller
        self.trim = self.ap.tr
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.max_steps = int(episode_s / CTRL_DT)
        self.sim: Optional[Fl.Sim] = None
        self._rng = np.random.default_rng()

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = self._rng
        if self.randomize:
            w = r.uniform(0, self.wind_max)
            ang = r.uniform(0, 2 * math.pi)
            wind = Fl.Wind(mean=(w * math.cos(ang), w * math.sin(ang), 0.0), turb=float(r.choice(self.turb_levels)),
                           seed=int(r.integers(1 << 30)))
            bank, pitch = r.uniform(-0.25, 0.25), self.trim["alpha"] + r.uniform(-0.1, 0.1)
            V0 = self.V + r.uniform(-2.0, 2.0)
            alt0 = self.alt0 + r.uniform(-15, 15)
            heading = r.uniform(-math.pi, math.pi)
        else:
            wind, bank, pitch, V0, alt0, heading = Fl.Wind(), 0.0, self.trim["alpha"], self.V, self.alt0, 0.0
        self.sim = Fl.Sim(self.veh, wind, dt=CTRL_DT / SUBSTEPS)
        pos = (0.0, -self.radius) if self.task == "orbit" else (0.0, 0.0)
        self.sim.reset(alt=alt0, airspeed=V0, heading=heading if self.task != "orbit" else 0.0, pos=pos, pitch=pitch, bank=bank)
        self.steps = 0
        self.energy0 = self.sim.used_wh()
        self._last = self.sim.obs()
        self.ap.int_h = 0.0           # the reference autopilot's integrators belong to the episode
        self.ap.int_V = 0.0
        return self._obs(self._last), {"wind": wind.mean, "turb": wind.turb}

    def _desired(self, o):
        """(course, bank feed-forward, cross-track error) of the task."""
        if self.task == "orbit":
            c, ff = AP.orbit_course(o, (0.0, 0.0), self.radius, +1)
            e = math.hypot(o["pos"][0], o["pos"][1]) - self.radius
            return c, ff, e
        if self.task == "line":
            c, e, _ = AP.line_course(o, (0.0, 0.0), (6000.0, 0.0))
            return c, 0.0, e
        return o["course"], 0.0, 0.0

    def _obs(self, o):
        c, _, e = self._desired(o)
        ce = AP.wrap(c - o["course"])
        t = self.trim
        return np.array([(self.alt0 - o["alt"]) / 20.0, (self.V - o["airspeed"]) / 5.0, math.sin(o["phi"]), math.cos(o["phi"]),
                         math.sin(o["theta"]), math.cos(o["theta"]), math.sin(ce), math.cos(ce), o["p"] / 2.0, o["q"] / 2.0, o["r"] / 2.0,
                         o["alpha"], o["beta"], e / 50.0, ce, o["surf"][0] / Fl.DEFL_MAX, o["surf"][1] / Fl.DEFL_MAX, o["surf"][2] / Fl.DEFL_MAX,
                         o["soc"], o["n"] - 1.0], np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        t = self.trim
        ctl = Fl.Controls(da=a[0] * Fl.DEFL_MAX, de=t["de"] + a[1] * Fl.DEFL_MAX, dr=a[2] * Fl.DEFL_MAX,
                          thr=float(np.clip(t["throttle"] + 0.5 * a[3], 0.0, 1.0)))
        sim = self.sim
        for _ in range(SUBSTEPS):
            o = sim.step(ctl, CTRL_DT / SUBSTEPS)
        self.steps += 1
        c, _, e = self._desired(o)
        eh, ev = o["alt"] - self.alt0, o["airspeed"] - self.V
        reward = 0.1 - 0.1 * (min(abs(eh) / 10.0, 3.0) + min(abs(ev) / 3.0, 3.0) + min(abs(e) / 40.0, 3.0))
        stalled = abs(o["alpha"]) > self.veh.tab.alpha_stall * 1.15
        left = abs(eh) > 80
        terminated = bool(sim.crashed or sim.structural_fail or stalled or left or sim.depleted())
        if sim.crashed or sim.structural_fail or stalled or left:
            reward -= 100.0
        truncated = self.steps >= self.max_steps
        info = {"alt_err": eh, "speed_err": ev, "track_err": e, "wh": sim.used_wh() - self.energy0, "power": o["power"],
                "max_n": sim.max_n, "crashed": sim.crashed, "structural": sim.structural_fail, "stalled": stalled}
        self._last = o
        return self._obs(o), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- reference policy
    def expert_action(self) -> np.ndarray:
        """The design-aware cascaded autopilot, expressed as an action in [-1, 1] (a strong baseline, and demonstrations)."""
        o = self._last
        chi, ff, _ = self._desired(o)
        ctl = self.ap.update(o, AP.Command(course=chi, alt=self.alt0, airspeed=self.V, phi_ff=ff), CTRL_DT)
        t = self.trim
        return np.clip([ctl.da / Fl.DEFL_MAX, (ctl.de - t["de"]) / Fl.DEFL_MAX, ctl.dr / Fl.DEFL_MAX, (ctl.thr - t["throttle"]) / 0.5],
                       -1, 1).astype(np.float32)


# ---------------------------------------------------------------- design environment
# parameter space of the parametric Falcon builder: name -> (low, high). Choices are indices into the catalog.
FALCON_SPACE = {
    "wing.span_mm": (1500.0, 2800.0),
    "wing.root_chord_mm": (170.0, 340.0),
    "wing.tip_chord_mm": (40.0, 170.0),
    "wing.break_y_mm": (300.0, 800.0),
    "wing.dihedral_deg": (0.0, 5.0),
    "htail.area_m2": (0.035, 0.14),
    "htail.arm_mm": (500.0, 1000.0),
    "vtail.area_m2": (0.02, 0.07),
    "fuselage.wall_mm": (1.2, 3.0),
}
FALCON_CHOICES = {
    "propulsion.prop": ["APC 9x6E", "APC 10x7E", "APC 11x8E", "APC 12x8E", "APC 13x8E"],
    "propulsion.motor": ["BrotherHobby Avenger 2816 620KV", "BrotherHobby Avenger 2816 900KV", "BrotherHobby Avenger 2816 1050KV"],
    "battery.part_number": ["6S2P Molicel P42A Li-ion", "6S1P Molicel P42A Li-ion", "6S 5000mAh 30C LiPo"],
}


class ParamCoder:
    """Maps a normalised action vector in [-1, 1] to builder parameters and back."""

    def __init__(self, space: Optional[dict] = None, choices: Optional[dict] = None):
        self.space = dict(space or FALCON_SPACE)
        self.choices = dict(choices or FALCON_CHOICES)
        self.names = list(self.space) + list(self.choices)

    def decode(self, a) -> dict:
        a = np.clip(np.asarray(a, float), -1, 1)
        sets, i = {}, 0
        for k, (lo, hi) in self.space.items():
            sets[k] = round(float(lo + (a[i] + 1) / 2 * (hi - lo)), 4)
            i += 1
        for k, opts in self.choices.items():
            sets[k] = opts[min(int((a[i] + 1) / 2 * len(opts)), len(opts) - 1)]
            i += 1
        return sets

    def encode(self, params: dict) -> np.ndarray:
        out = []
        for k, (lo, hi) in self.space.items():
            out.append(2 * (self._get(params, k) - lo) / (hi - lo) - 1)
        for k, opts in self.choices.items():
            v = self._get(params, k)
            j = opts.index(v) if v in opts else 0
            out.append(2 * (j + 0.5) / len(opts) - 1)
        return np.array(out, np.float32)

    @staticmethod
    def _get(d, path):
        cur = d
        for k in path.split("."):
            cur = cur[k]
        return cur


class DesignEnv(gym.Env):
    """Design as a Gymnasium problem.

    action: Box(-1, 1, (n_params + n_choices,)): every parametric knob, rescaled; choices are binned.
    observation: the current normalised parameters, then [score/baseline, gate checks passed fraction, endurance, mass,
                 static margin, last step accepted, last step failed gate, steps left].
    reward: (new score - previous best)/baseline when the gate passes (also negative if it is worse), -1 when the gate fails.
            The design only advances to a candidate that passes the gate and is not worse (the same rule as `dronebench try`).
    Each step builds the aircraft and flies it (about 20 s with `fidelity='flown'`, 5 s with 'analytic').
    """
    metadata = {"render_modes": []}

    def __init__(self, fixture: str = "falcon_v2", preset: str = "endurance", max_steps: int = 20, fidelity: str = "flown",
                 workdir: Optional[str] = None, space: Optional[dict] = None, choices: Optional[dict] = None):
        super().__init__()
        self.fixture, self.preset, self.max_steps, self.fidelity = fixture, preset, max_steps, fidelity
        self.coder = ParamCoder(space, choices)
        self.space, self.choices, self.names = self.coder.space, self.coder.choices, self.coder.names
        n = len(self.names)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n + 8,), np.float32)
        self._own = workdir is None
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="dbenv_"))
        self.ws: Optional[Path] = None
        self.n = 0

    # ---- parameter coding
    def decode(self, a) -> dict:
        return self.coder.decode(a)

    def encode(self, params: dict) -> np.ndarray:
        return self.coder.encode(params)

    # ---- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        from . import workspace as W
        import yaml
        if self.ws and self._own:
            shutil.rmtree(self.ws, ignore_errors=True)
        self.ws = self.workdir / f"ep{abs(hash((seed, id(self)))) % 10**8}"
        W.init(self.ws, self.fixture, preset=self.preset, steps=self.max_steps, force=True, necessity=False, fidelity=self.fidelity)
        self.params = yaml.safe_load((self.ws / "params.yaml").read_text())
        self.n = 0
        self.last = {"accepted": True, "failed": False}
        return self._obs(), {"baseline": W.meta(self.ws)["baseline"]}

    def _obs(self):
        from . import workspace as W
        m = W.meta(self.ws)
        rep = W.current_report(self.ws)
        ac = rep.get("aircraft") or {}
        checks = (rep.get("gate") or {}).get("checks", [])
        frac = sum(1 for c in checks if c.get("pass")) / max(len(checks), 1)
        base = max(m["baseline"], 1e-9)
        extra = [m["best"] / base, frac, (ac.get("endurance_min") or 0) / 100.0, (ac.get("mass_g") or 0) / 3000.0,
                 ac.get("static_margin") or 0.0, float(self.last["accepted"]), float(self.last["failed"]), (self.max_steps - self.n) / self.max_steps]
        return np.concatenate([self.encode(self.params), np.array(extra, np.float32)]).astype(np.float32)

    def step(self, action):
        from . import workspace as W
        sets = self.decode(action)
        before = W.meta(self.ws)["best"]
        out = W.build_step(self.ws, sets, note="rl step", agent="rl", necessity=False)
        self.n += 1
        failed = out["reason"] == "gate failed" or bool(out["errors"])
        if out["accepted"]:
            import yaml
            self.params = yaml.safe_load((self.ws / "params.yaml").read_text())
            reward = (out["score"] - before) / max(W.meta(self.ws)["baseline"], 1e-9)
        elif failed:
            reward = -1.0
        else:
            reward = (out["score"] - before) / max(W.meta(self.ws)["baseline"], 1e-9)   # a worse design: negative
        self.last = {"accepted": out["accepted"], "failed": failed}
        truncated = self.n >= self.max_steps
        return self._obs(), float(reward), False, truncated, {"outcome": out, "sets": sets, "best": W.meta(self.ws)["best"]}

    def close(self):
        if self._own and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)


gym.register(id="DroneBench/Flight-v0", entry_point="dronebench.envs:FlightEnv")
gym.register(id="DroneBench/Design-v0", entry_point="dronebench.envs:DesignEnv")
