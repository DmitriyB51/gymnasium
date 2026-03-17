# pendulum_sac.py
# Solve Pendulum-v1 using SAC (Soft Actor-Critic)
# The agent learns by itself — no hand-coded rules

import gymnasium as gym
from stable_baselines3 import SAC


env = gym.make("Pendulum-v1")


model = SAC(
    "MlpPolicy",          # MLP = simple neural network (not CNN or transformer)
    env,
    learning_rate=0.001,   # how fast the networks update
    batch_size=256,        # how many experiences to learn from at once
    verbose=1,             # print training progress
)



model.learn(total_timesteps=50_000)


model.save("pendulum_sac_model")




env.close()
env = gym.make("Pendulum-v1", render_mode="human")

for game in range(5):
    state, info = env.reset()
    total_reward = 0

    while True:
        # model.predict gives us the learned action — no randomness
        action, _ = model.predict(state, deterministic=True)

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Game {game + 1}: reward = {total_reward:.0f}")

env.close()
