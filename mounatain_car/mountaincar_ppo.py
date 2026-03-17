# mountaincar_ppo.py
# Solve MountainCarContinuous-v0 using PPO
# PPO is also the algorithm you'll use for the Go2 later

import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("MountainCarContinuous-v0")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=1024,              # collect this many steps before each update
    batch_size=64,
    n_epochs=10,               # train on collected data 10 times
    policy_kwargs=dict(net_arch=[64, 64]),  # smaller network — faster training
    verbose=1,
)

print("Training...\n")
model.learn(total_timesteps=100_000)

model.save("mountaincar_ppo_model")

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
