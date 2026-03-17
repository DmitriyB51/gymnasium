# mountaincar_v2.py
# MountainCar solved with SAC + reward shaping + proper settings

import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        position = state[0]
        velocity = state[1]

        # reward based on height — higher position = bigger reward
        # position ranges from -1.2 to 0.6, goal is at 0.45
        reward += position + 1.2  # shifts to 0.0 - 1.8 range, always positive

        return state, reward, terminated, truncated, info

env = RewardShapingWrapper(gym.make("MountainCarContinuous-v0"))

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    batch_size=256,
    learning_starts=1000,      # collect 1000 random steps before training starts
    ent_coef="auto",           # let SAC auto-tune exploration this time
    policy_kwargs=dict(net_arch=[64, 64]),
    device="cpu",
    verbose=1,
)

print("Training...\n")
model.learn(total_timesteps=50_000)

model.save("mountaincar_v2_model")

# Test on ORIGINAL environment
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
