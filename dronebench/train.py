"""Train and evaluate a PPO flight controller on FlightEnv (stable-baselines3, optional dependency).

    python -m dronebench.train --design fixtures/falcon_v2 --task orbit --steps 600000 --out runs/ppo_orbit

Writes `learning.json` (evaluation return every `eval_every` steps, with the autopilot, zero-action and random
policies as reference lines) and `ppo.zip`. Evaluation uses fixed seeds, so curves from different runs compare.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def evaluate(env, policy, seeds=range(100, 108)):
    rets, ok = [], 0
    for s in seeds:
        o, _ = env.reset(seed=s)
        done, R = False, 0.0
        while not done:
            a = policy(o)
            o, r, term, trunc, info = env.step(a)
            R += r
            done = term or trunc
        rets.append(R)
        ok += int(not term)
    return float(np.mean(rets)), ok / len(list(seeds))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="fixtures/falcon_v2")
    ap.add_argument("--task", default="orbit", choices=("orbit", "line", "hold"))
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from .envs import FlightEnv

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mk = lambda: FlightEnv(a.design, task=a.task, episode_s=45.0)
    ev = mk()
    refs = {"autopilot": evaluate(ev, lambda o: ev.expert_action()),
            "zero": evaluate(ev, lambda o: np.zeros(4, np.float32)),
            "random": evaluate(ev, lambda o: ev.action_space.sample())}
    print("reference returns (mean, survival):", {k: (round(v[0], 1), v[1]) for k, v in refs.items()}, flush=True)
    venv = VecNormalize(DummyVecEnv([mk for _ in range(4)]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
                ent_coef=0.0, seed=a.seed, verbose=0, policy_kwargs={"net_arch": [96, 96]})
    curve, t0 = [], time.time()

    class Eval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = 0

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.next += a.eval_every
                venv.training = False
                pol = lambda o: model.predict(venv.normalize_obs(o), deterministic=True)[0]
                r, surv = evaluate(ev, pol)
                venv.training = True
                curve.append({"steps": int(self.num_timesteps), "return": r, "survival": surv, "minutes": round((time.time() - t0) / 60, 1)})
                print(curve[-1], flush=True)
                (out / "learning.json").write_text(json.dumps({"task": a.task, "design": a.design, "reference": refs, "curve": curve}, indent=1))
            return True

    model.learn(total_timesteps=a.steps, callback=Eval())
    model.save(out / "ppo.zip")
    venv.save(out / "vecnorm.pkl")


if __name__ == "__main__":
    main()
