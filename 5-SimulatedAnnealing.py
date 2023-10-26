import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

env_name = 'CartPole-v0'
env = gym.make(env_name)
# env = gym.wrappers.Monitor(env, "./gym-results", force=True)

def computeAction(observation, weights):
    evaluation = np.matmul(weights, observation)
    return 0 if evaluation < 0 else 1

# Initialise variables
history = []
weight_results = []
intermediate_history = []

nr_of_iterations = 1000
nr_episodes_per_iteration = 20
best_reward = 0

# Hill Climbing Action - add noise to the weights
climbing_history = []
climbing_weights = []
noise_scaling = 1e-3
#best_weights = 2 * np.random.rand(4) - 1 # generate random weights between -1 and 1

# Simulated annealing 
Temperature = 10000
cooling_rate = 0.5
spread = 0.1

scaling_history = []
scaling_weights = []

for _ in range(nr_of_iterations):
    weights = 2 * np.random.rand(4) - 1 # generate random weights between -1 and 1
    intermediate_history = []
    for _ in range(nr_episodes_per_iteration):
        observation = env.reset()
        # env.render()
        done = False
        total_rew = 0
        while not done:
            action = computeAction(observation, weights)
            observation, reward, done, info = env.step(action)
            total_rew = total_rew + reward
        intermediate_history.append(total_rew)
    average_reward = np.mean(intermediate_history)
    weight_results.append(weights)
    history.append(average_reward)
    if average_reward > best_reward:
        best_weights = weights
        best_reward = average_reward

for _ in range(nr_of_iterations):
    current_weights = best_weights + np.random.normal(loc=0, scale=spread)
    intermediate_history = []
    for _ in range(nr_episodes_per_iteration):
        observation = env.reset()
        done = False
        total_rew = 0
        while not done:
            action = computeAction(observation, current_weights)
            observation, reward, done, info = env.step(action)
            total_rew = total_rew + reward
        intermediate_history.append(total_rew)
    average_reward = np.mean(intermediate_history)
    scaling_weights.append(current_weights)
    scaling_history.append(average_reward)
    if average_reward >= best_reward:
        best_weights = current_weights
        best_reward = average_reward
    else:
        reward_difference = (average_reward - best_reward) # => -Delta (best_reward - average_reward = Delta)
        p = np.exp(reward_difference/Temperature)
        if np.random.rand() < p:
            best_weights = current_weights
            best_reward = average_reward
    Temperature = cooling_rate * Temperature

# Test the best weights
test_history = []
test_iterations = 1000
max_episodes = 200 # maximum reward for CartPole-v0

for _ in range(test_iterations):
    observation = env.reset()
    weights = best_weights
    total_rew = 0   
    for _ in range(max_episodes):
        # env.render()
        action = computeAction(observation, weights)
        observation, reward, done, info = env.step(action) # take the action
        total_rew = total_rew + reward # update the total reward
        if done:
            test_history.append(total_rew)
            total_rew = 0
            break

env.close()

# Convert the lists to dataframes
df_scaling = pd.DataFrame(np.row_stack(scaling_weights), columns=['cart pos','cart velocity','angle','angular velocity'])
df_scaling['history'] = scaling_history
df_scaling['value'] = np.where(df_scaling['history']>100, 1, 0)
print(df_scaling)

# Print the statistics
print("Statistics of the rewards per episode")
print("Average reward = ",df_scaling["history"].mean())
print("Standard Deviation of the reward = ",df_scaling["history"].std())
print("Best reward = ",df_scaling["history"].max())
print("Minimum reward = ",df_scaling["history"].min())
print("Best weights = ",best_weights)
# Show the distribution of the rewards or timesteps
sns.displot(data=scaling_history)
# plt.show()

# 3D Scatterplot of the weights
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # 1 by 1 by 1 grid (default), 1st subplot
cmap = matplotlib.colors.ListedColormap(['black', 'red'])
ax.scatter(df_scaling['cart velocity'], df_scaling['angle'], df_scaling['angular velocity'], c=df_scaling['value'], cmap=cmap, linewidth=0.5)
ax.set_xlabel('Cart Velocity')
ax.set_ylabel('Angle')
ax.set_zlabel('Angular Velocity')
ax.set_title('Weights and Rewards')
# plt.show()

# Print the statistics
print("Statistics of the rewards per test episode")
print("Average reward = ",np.mean(test_history))
print("Standard Deviation of the reward = ",np.std(test_history))
print("Best reward = ",np.max(test_history))
print("Minimum reward = ",np.min(test_history))
# Show the distribution of the rewards or timesteps
sns.displot(data=test_history)
plt.show()