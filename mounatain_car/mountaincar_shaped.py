# mountaincar_shaped.py
# MountainCar with reward shaping — reward the agent for climbing higher

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# Custom wrapper that adds reward for reaching higher positions
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_position = -1.2  # track highest position this episode (start at leftmost)

    def reset(self, **kwargs):
        self.max_position = -1.2
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        position = state[0]
        velocity = state[1]

        # reward for reaching new heights (small bonus, not 10)
        if position > self.max_position:
            reward += (position - self.max_position)
            self.max_position = position

        # small reward for having velocity (building momentum)
        reward += 0.01 * abs(velocity)

        return state, reward, terminated, truncated, info

env = RewardShapingWrapper(gym.make("MountainCarContinuous-v0"))

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    policy_kwargs=dict(net_arch=[64, 64]),
    device="cpu",
    verbose=1,
)

print("Training with reward shaping...\n")
model.learn(total_timesteps=100_000)

model.save("mountaincar_shaped_model")

# Watch with the ORIGINAL environment (no shaping) to see real score
env.close()
env = gym.make("MountainCarContinuous-v0", render_mode="human")

for game in range(5):
    state, info = env.reset()
    total_reward = 0

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Game {game + 1}: reward = {total_reward:.0f}")

env.close()
