import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Hopper-v4")

model = PPO(
    "MlpPolicy",
    env, 
    learning_rate=0.0003,
    n_steps = 2048,
    batch_size=64, 
    n_epochs=10,
    policy_kwargs=dict(net_arch=[256,256]),
    device="cpu",
    verbose=1,

)

model.learn(total_timesteps=500_000)

model.save("hopper_model")

env.close()
