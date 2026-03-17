# mountaincar_sac.py
# Solve MountainCarContinuous-v0 using SAC

import gymnasium as gym
from stable_baselines3 import SAC


env = gym.make("MountainCarContinuous-v0")


model = SAC(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    batch_size=256,
    verbose=1,
)



model.learn(total_timesteps=100_000)

model.save("mountaincar_sac_model")



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
