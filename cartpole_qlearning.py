# cartpole_qlearning.py
# The agent learns the strategy by itself using a Q-table

import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")



NUM_BINS = 10
BINS = [
    np.linspace(-2.4, 2.4, NUM_BINS),     # cart position
    np.linspace(-2.0, 2.0, NUM_BINS),      # cart velocity
    np.linspace(-0.25, 0.25, NUM_BINS),    # pole angle
    np.linspace(-2.0, 2.0, NUM_BINS),      # pole angular velocity
]

def discretize(state):
    """Convert continuous state into bin indices"""
    discrete = []
    for i, val in enumerate(state):
        # np.digitize tells us which bin this value falls into
        bin_index = np.digitize(val, BINS[i]) - 1
        # clamp to valid range
        bin_index = min(NUM_BINS - 1, max(0, bin_index))
        discrete.append(bin_index)
    return tuple(discrete)

# q-table
# Shape: 10 x 10 x 10 x 10 x 2
# (10 bins for each of 4 state values, 2 possible actions)
# Each entry stores: "how good is it to take this action in this state?"
q_table = np.zeros([NUM_BINS] * 4 + [2])


LEARNING_RATE = 0.1      # how much we update the table each step
DISCOUNT = 0.99          # how much we care about future rewards
EPISODES = 10000         # how many games to play
EPSILON_START = 1.0      # start fully random (exploring)
EPSILON_END = 0.01       # end mostly greedy (exploiting what we learned)
EPSILON_DECAY = EPSILON_START / (EPISODES * 0.8)  # decay over 80% of training

epsilon = EPSILON_START

# Training loop
best_score = 0
scores = []

for episode in range(EPISODES):
    state, info = env.reset()
    discrete_state = discretize(state)
    score = 0

    while True:
        # Epsilon-greedy: sometimes explore (random), sometimes exploit (best known)
        if np.random.random() < epsilon:
            action = env.action_space.sample()         # explore: try random action
        else:
            action = np.argmax(q_table[discrete_state])  # exploit: pick best action

        # Take the action
        new_state, reward, terminated, truncated, info = env.step(action)
        new_discrete_state = discretize(new_state)
        score += reward

  
        if not terminated:
            best_future = np.max(q_table[new_discrete_state])
            current = q_table[discrete_state + (action,)]
            q_table[discrete_state + (action,)] += LEARNING_RATE * (
                reward + DISCOUNT * best_future - current
            )
        else:
            
            q_table[discrete_state + (action,)] += LEARNING_RATE * (
                reward - q_table[discrete_state + (action,)]
            )

        discrete_state = new_discrete_state

        if terminated or truncated:
            break

    
    epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)

    scores.append(score)
    if score > best_score:
        best_score = score

   
    if (episode + 1) % 500 == 0:
        avg = np.mean(scores[-500:])
        print(f"Episode {episode + 1:>5} | avg score: {avg:>6.1f} | best: {int(best_score):>3} | epsilon: {epsilon:.2f}")






print("\n--- Watching trained agent ---\n")
env.close()
env = gym.make("CartPole-v1", render_mode="human")

for game in range(5):
    state, info = env.reset()
    discrete_state = discretize(state)
    score = 0

    while True:
        action = np.argmax(q_table[discrete_state])  # no randomness, pure exploit
        state, reward, terminated, truncated, info = env.step(action)
        discrete_state = discretize(state)
        score += reward

        if terminated or truncated:
            break

    print(f"Game {game + 1}: {int(score)} points")

env.close()
