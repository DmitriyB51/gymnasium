# pendulum_random.py
# Step 1: See the environment with a random agent

import gymnasium as gym
import time

env = gym.make("Pendulum-v1", render_mode="human")

for game in range(3):
    state, info = env.reset()
    total_reward = 0

    for step in range(200):
        # state[0] = cos(angle), state[1] = sin(angle), state[2] = angular velocity
        # action is continuous: torque from -2.0 to +2.0
        action = env.action_space.sample()  # random torque

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        time.sleep(0.02)

        if terminated or truncated:
            break

    print(f"Game {game + 1}: reward = {total_reward:.0f}")

env.close()
