# my_first_agent.py
# v2: uses angular velocity to predict where the pole is heading

import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

for game in range(5):
    state, info = env.reset()
    score = 0

    while True:
        pole_angle = state[2]
        pole_velocity = state[3]

        # predict where the pole is heading, not just where it is now
        if pole_angle + pole_velocity > 0:
            action = 1  # push right
        else:
            action = 0  # push left

        state, reward, terminated, truncated, info = env.step(action)
        score += reward

        time.sleep(0.05)

        if terminated or truncated:
            break

    print(f"Game {game + 1}: scored {int(score)} points")

env.close()
print("Done!")