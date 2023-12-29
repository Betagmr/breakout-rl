import numpy as np
import torch
from matplotlib import pyplot as plt

from src.breakout import BreakoutEnv
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank, seed=0):
    def _init():
        env = BreakoutEnv(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def train():
    n_env = 20 
    env = SubprocVecEnv([make_env(i) for i in range(n_env)])

    model = PPO("MlpPolicy", env, verbose=1)
    model.load("dqn_breakout-300k", device="cuda")
    model.learn(total_timesteps=700_000, progress_bar=True, )
    model.save("dqn_breakout-1000k")


def test_model(model_name):
    env = BreakoutEnv(render_mode="human")
    model = PPO.load(model_name, env=env)

    obs, info = env.reset()

    while True:
        action = model.predict(obs)[0] 
        obs, reward, done, truncate, info = env.step(action)
        env.render()

        if done:
            break

if __name__ == "__main__":
    train()