import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("legged/hopper_model")

env = gym.make("Hopper-v4", render_mode="human")

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
