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
    return env.action_space.sample()

for _ in range(nr_of_iterations):
    # Reset the environment to get the initial state 
    # and get the initial observation with angle = observation[2]
    observation = env.reset()
    total_rew = 0   
    for _ in range(nr_episodes_per_iteration): # replace '_' with 't' to see the time steps
        #env.render()
        action = computeAction(observation)
        observation, reward, done, info = env.step(action) # take the action
        total_rew = total_rew + reward # update the total reward
        if done:
            history.append(total_rew)
            break
env.close()

# print the statistics
print("Average reward = ",np.mean(history))
print("Standard Deviation of the reward = ",np.std(history))
print("Maximum reward = ",np.max(history))
print("Minimum reward = ",np.min(history))
print("Total number of steps = ",np.sum(history)+1)

# Show the distribution of the rewards or timesteps
df = pd.DataFrame(history, columns=['Rewards'])
sns.displot(data=df, legend=False)
plt.title("CartPole-V0 Random Action")
plt.subplots_adjust(top=0.95)
plt.figtext(0.6, 0.6, df.describe().loc[['count','mean','std']].to_string())
plt.show()