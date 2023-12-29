import gymnasium as gym
import numpy as np
from gymnasium import spaces
from random import randint
from src.utils import get_map, get_platform_cords, get_ball_position
from src.const import ACTION_FIRE_BALL, ACTION_LEFT, ACTION_RIGHT, ACTION_NO_MOVE


class BreakoutEnv(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        self.env = gym.make(
            "BreakoutNoFrameskip-v4",
            render_mode=render_mode,
        )
        self.env.metadata["render_fps"] = 60
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32)

        # Env variables
        self.ball_previous_y = None

    def step(self, action):
        if action == 0:
            action = ACTION_LEFT
        elif action == 1:
            action = ACTION_RIGHT
        elif action == 2:
            action = ACTION_NO_MOVE

        if self.n_steps > 1000 and not self.is_down:
            action = randint(0, 2)
            print("No move")

        obs_current, reward, done, truncated, info = self.env.step(action)

        if reward > 0:
            self.n_steps = 0

        self.previous_state = self.current_state
        self.current_state = get_map(obs_current)

        if info["lives"] != self.n_lives:
            self.is_down = False
            self.n_steps = 0
            self.n_lives = info["lives"]

        if self.n_lives == 4:
            done = True

        state = self.get_state(self.current_state, self.previous_state, done)
        reward = self.get_reward(self.current_state, self.previous_state, done)
        self.n_steps += 1

        return state, reward, done, truncated, info

    def reset(self, seed=None):
        obs_prev, info = self.env.reset(seed=seed)
        obs_current, *_, info = self.env.step(ACTION_FIRE_BALL)

        self.n_steps = 0
        self.n_lives = 5
        self.is_down = False
        self.previous_state = get_map(obs_prev)
        self.current_state = get_map(obs_current)

        state = self.get_state(self.current_state, self.previous_state)

        return state, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_state(self, current_state, previous_state, done=False):
        """
        Returns a state vector:
            0: Ball is in left side
            1: Ball is in left inside the platform
            2: Ball is in center of the platform
            3: Ball is in right inside the platform
            4: Ball is in right side
        """
        if done:
            return np.array([0, 0, 0], dtype=np.int32)

        position = get_ball_position(current_state, previous_state)

        while position is None:
            current_state, *_ = self.env.step(ACTION_FIRE_BALL)
            self.current_state = get_map(current_state)
            position = get_ball_position(self.current_state, previous_state)

        ball_x, _ = position
        platform_left, platform_right, platform_center = get_platform_cords(
            current_state
        )

        return np.array(
            [
                ball_x < platform_left,
                platform_left <= ball_x <= platform_center - 3,
                platform_center - 3 < ball_x < platform_center + 4,
                platform_center + 4 <= ball_x <= platform_right,
                ball_x > platform_right,
            ],
            dtype=np.int32,
        )

    def get_reward(self, current_state, previous_state, done):
        if done:
            self.is_down = False
            return -10

        _, y_position = get_ball_position(current_state, previous_state)

        if not self.is_down and 143 < y_position < 145:
            self.is_down = True

        if self.is_down and 135 < y_position < 143:
            self.is_down = False
            return 10

        return 0
