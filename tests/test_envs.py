import numpy as np
import pytest

import dronebench.envs as E


def test_param_coder_roundtrip():
    c = E.ParamCoder()
    params = {"wing": {"span_mm": 2093, "root_chord_mm": 260, "tip_chord_mm": 51, "break_y_mm": 602, "dihedral_deg": 0.5},
              "htail": {"area_m2": 0.075, "arm_mm": 750}, "vtail": {"area_m2": 0.04}, "fuselage": {"wall_mm": 2.4},
              "propulsion": {"prop": "APC 11x8E", "motor": "BrotherHobby Avenger 2816 620KV"},
              "battery": {"part_number": "6S2P Molicel P42A Li-ion"}}
    a = c.encode(params)
    assert a.shape == (len(c.names),) and np.all(np.abs(a) <= 1.0)
    d = c.decode(a)
    assert abs(d["wing.span_mm"] - 2093) < 1.0 and d["propulsion.prop"] == "APC 11x8E"
    assert d["battery.part_number"] == "6S2P Molicel P42A Li-ion"


@pytest.fixture(scope="module")
def env():
    return E.FlightEnv("fixtures/trainer", task="orbit", episode_s=20.0)


def test_flight_env_spaces_and_step(env):
    o, info = env.reset(seed=3)
    assert o.shape == env.observation_space.shape and o.dtype == np.float32
    o2, r, term, trunc, info = env.step(np.zeros(4, np.float32))
    assert o2.shape == o.shape and isinstance(r, float) and "track_err" in info


def test_autopilot_survives_and_zero_action_does_not(env):
    def episode(policy, seed):
        env.reset(seed=seed)
        R, done, term = 0.0, False, False
        while not done:
            o, r, term, trunc, info = env.step(policy())
            R += r
            done = term or trunc
        return R, term
    exp = [episode(env.expert_action, s) for s in range(12)]
    zero = [episode(lambda: np.zeros(4, np.float32), s) for s in range(12)]
    assert all(not t for _, t in exp), "the design-aware autopilot must fly every randomised episode"
    assert np.mean([r for r, _ in exp]) > np.mean([r for r, _ in zero]) + 20


def test_flight_env_is_deterministic_for_a_seed(env):
    def roll(seed):
        env.reset(seed=seed)
        return [env.step(np.array([0.1, 0.0, 0.0, 0.0], np.float32))[0] for _ in range(50)]
    a, b = roll(5), roll(5)
    assert np.allclose(a[-1], b[-1])


def test_gym_registration():
    import gymnasium as gym
    e = gym.make("DroneBench/Flight-v0", design="fixtures/trainer", episode_s=5.0)
    o, _ = e.reset(seed=0)
    assert o.shape == (20,)
