import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

env_name = 'CartPole-v0'
env = gym.make(env_name)
# env = gym.wrappers.Monitor(env, "./gym-results", force=True)

# Initialise variables
history = []
weight_results = []
intermediate_history = []
average_reward = 0
nr_of_iterations = 1000
nr_episodes_per_iteration = 20
best_reward = 0
total_rew = 0
done = False
weights = 2 * np.random.rand(4) - 1 # generate random weights between -1 and 1
best_weights = weights

def computeAction(observation, weights):
    evaluation = np.matmul(weights, observation)
    return 0 if evaluation < 0 else 1

for _ in range(nr_of_iterations):
    observation = env.reset()
    weights = 2 * np.random.rand(4) - 1 # generate random weights between -1 and 1
    intermediate_history = []
    for episode in range(nr_episodes_per_iteration):
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
    if average_reward> best_reward:
        best_weights = weights
        best_reward = average_reward
    # print("Iteration = ", iteration, " Total reward = ",total_rew, " Best reward = ",best_reward, " Weights = ",weights)

# Test the best weights
test_history = []
test_iterations = 1000
max_episodes = 200 # maximum reward for CartPole-v0

for _ in range(test_iterations):
    observation = env.reset()
    weights = best_weights 
    for episode in range(max_episodes):
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
df = pd.DataFrame(np.row_stack(weight_results), columns=['cart pos','cart velocity','angle','angular velocity'])
df['history'] = history
df['value'] = np.where(df['history']>100, 1, 0)
print(df)

# Print the statistics
print("Statistics of the rewards per episode")
print("Average reward = ",df["history"].mean())
print("Standard Deviation of the reward = ",df["history"].std())
print("Best reward = ",df["history"].max())
print("Minimum reward = ",df["history"].min())
print("Best weights = ",best_weights)
# Show the distribution of the rewards or timesteps
sns.displot(data=history)
# plt.show()

# 3D Scatterplot of the weights
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # 1 by 1 by 1 grid (default), 1st subplot
cmap = matplotlib.colors.ListedColormap(['black', 'red'])
ax.scatter(df['cart velocity'], df['angle'], df['angular velocity'], c=df['value'], cmap=cmap, linewidth=0.5)
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