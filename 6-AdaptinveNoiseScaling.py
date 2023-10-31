import gym
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

env_name = 'CartPole-v0'
methodname = 'Adaptive Noise Scaling'
env = gym.make(env_name)
# env = gym.wrappers.Monitor(env, "./gym-results", force=True)

def computeAction(observation, weights):
    evaluation = np.matmul(weights, observation)
    return 0 if evaluation < 0 else 1

# Initialise variables
history = []
weight_results = []
episode_history = []
climbing_history = []
climbing_weights = []
best_reward = 0

# Set the parameters
nr_of_iterations = 1000
nr_episodes_per_iteration = 20

# Hill Climbing Action - add noise to the weights
noise_scaling = 0.3

# Simulated annealing 
Temperature = 750000
cooling_rate = 0.85

# Adaptive noise scaling
spread = 0.4
min_spread = 0.1
max_spread = 0.8

start = timer()
best_weights = 2 * np.random.rand(4) - 1 # generate dummy random weights between -1 and 1
# Simulated annealing + adaptive noise scaling
for _ in range(nr_of_iterations):
    current_weights = best_weights + noise_scaling * np.random.normal(loc=0, scale=spread, size=4)
    episode_history = []
    for _ in range(nr_episodes_per_iteration):
        observation = env.reset()
        done = False
        total_rew = 0
        while not done:
            action = computeAction(observation, current_weights)
            observation, reward, done, info = env.step(action)
            total_rew = total_rew + reward
        episode_history.append(total_rew)
    average_reward = np.mean(episode_history)
    climbing_weights.append(current_weights)
    climbing_history.append(average_reward)
    if average_reward >= best_reward:
        best_weights = current_weights
        best_reward = average_reward
        spread = max(spread/2, min_spread)
    else:
        reward_difference = (average_reward - best_reward) # => -Delta (best_reward - average_reward = Delta)
        p = np.exp(reward_difference/Temperature)
        if np.random.rand() < p:
            best_weights = current_weights
            best_reward = average_reward
            spread = min(spread*2, max_spread)
    Temperature = cooling_rate * Temperature
end = timer()

# calculate runtime of method
runningtime = "Running time: "+str(timedelta(seconds=end-start))
print(runningtime)

# Test the best weights
test_history = []
test_episodes = 1000

for _ in range(test_episodes):
    observation = env.reset()
    weights = best_weights
    total_rew = 0   
    done = False
    while not done:
        action = computeAction(observation, weights)
        observation, reward, done, info = env.step(action) # take the action
        total_rew = total_rew + reward # update the total reward
    test_history.append(total_rew)

env.close()

# Convert the lists to dataframes
df_climbing = pd.DataFrame(np.row_stack(climbing_weights), columns=['cart position','cart velocity','angle','angular velocity'])
df_climbing['history'] = climbing_history
df_climbing['value'] = np.where(df_climbing['history']<100, 0, 
                            np.where(df_climbing['history']<180, 0.5, 1))


best_weights = np.round(best_weights, 4)
best = "Best weights = "+" ".join(str(x) for x in best_weights)
plottitle = env_name+" "+methodname+" with "+best

# Show the distribution of the rewards or timesteps
# Show 4 plots in one figure - Histogram Method, 3D Scatterplot weights, Lineplot Reward and Histogram Test rewards
fig, ax = plt.subplots(2,2)
fig.suptitle(plottitle, x=0.5, y=0.98)
ax[0,0].set_title("Histogram of Rewards for method")
ax[0,1].set_title("3D Scatterplot of Weights and Rewards")
ax[1,0].set_title("Lineplot of Rewards per iteration")
ax[1,1].set_title("Histogram of Rewards for test episodes")
# Histogram of rewards for method [0,0]
df_rewards = pd.DataFrame(climbing_history, columns=['Rewards'])
fig1 = sns.histplot(data=df_rewards, ax=ax[0,0], legend=False)
ax[0,0].text(0.1, 0.7, df_rewards.describe().loc[['count','mean','std']].to_string(), transform=ax[0,0].transAxes)
# 3D Scatterplot of the weights [0,1]
ax3D = fig.add_subplot(2,2,2, projection='3d')
cmap = matplotlib.colors.ListedColormap(['black', 'orange', 'red'])
ax3D.scatter(df_climbing['cart position'], df_climbing['angle'], df_climbing['angular velocity'], c=df_climbing['value'], cmap=cmap, linewidth=0.5)
ax3D.set_xlabel('Cart Position')
ax3D.set_ylabel('Angle')
ax3D.set_zlabel('Angular Velocity')
ax[0,1].set_axis_off()
ax[0,1].text(0.1, 0.7, "<100", color="black", transform=ax[0,1].transAxes)
ax[0,1].text(0.1, 0.8, "100><180", color="orange", transform=ax[0,1].transAxes)
ax[0,1].text(0.1, 0.9, ">180", color="red", transform=ax[0,1].transAxes)
# Lineplot of rewards per episode [1,0]
fig3 = sns.lineplot(data=df_climbing, x=df_climbing.index, y='history', ax=ax[1,0])
ax[1,0].set_xlabel('Iterations')
ax[1,0].set_ylabel('Reward')
ax[1,0].text(0.5, 0.1, runningtime, transform=ax[1,0].transAxes)
# Histogram of rewards for test episodes [1,1]
df_test = pd.DataFrame(test_history, columns=['Rewards'])
fig4 = sns.histplot(data=df_test, ax=ax[1,1], legend=False)
ax[1,1].text(0.1, 0.7, df_test.describe().loc[['count','mean','std']].to_string(), transform=ax[1,1].transAxes)
plt.show()