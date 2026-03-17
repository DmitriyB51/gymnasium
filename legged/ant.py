import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Ant-v4")

model = PPO(
    "MlpPolicy",
    env, 
    learning_rate=0.0003,
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    policy_kwargs=dict(net_arch=[256,256]),
    device="cpu",
    verbose=1,
)

model.learn(total_timesteps=3_000_000)

model.save("ant_model")

env.close()
