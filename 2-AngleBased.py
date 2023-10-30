import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialise counters
history = []
nr_of_iterations = 1000
nr_episodes_per_iteration = 200

# Create the environment
env = gym.make('CartPole-v0')
# Store every episode as a video in gym-results folder
# env = gym.wrappers.Monitor(env, "./gym-results", force=True)
def computeAction(observation):
    return 0 if observation[2] < 0 else 1

for _ in range(nr_of_iterations):
    # Reset the environment to get the initial state 
    # and get the initial observation with angle = observation[2]
    observation = env.reset()
    total_rew = 0
    for _ in range(nr_episodes_per_iteration):
        env.render()
        action = computeAction(observation)
        observation, reward, done, info = env.step(action)
        total_rew = total_rew + reward
        if done:
            history.append(total_rew)
            break
env.close()

# Print the statistics
print("Average reward = ",np.mean(history))
print("Standard Deviation of the reward = ",np.std(history))
print("Maximum reward = ",np.max(history))
print("Minimum reward = ",np.min(history))
print("Total reward = ",np.sum(history))

# Show the distribution of the rewards or timesteps
df = pd.DataFrame(history, columns=['Rewards'])
sns.displot(data=df)
plt.title("CartPole-V0 Angle Based Action")
plt.show()