# RL_1_CartPole
## Solving the CartPole-v0 OpenAI
### 1.0 Introduction
- **Episode** = A sequence of interactions between an agent and its environment, starting from an initial state and ening at a terminal state.
- The agent and the environment interact at each timestep
- At each timestep, the agent receives an **observation** and a **reward** from the environment
- The agent chooses an **action** based on the observation
- The environment transitions to a new state and emits a new observation and reward
- The episode ends when a terminal state is reached
- The agent learns a policy, which is a mapping from observations to actions
- The goal of the agent is to maximize the total reward over the episode

### 1.1 Random action based control
![Histogram of Random action based control](gym-results/1-Histogram_Random_Action.png)

### 1.2 Angle based action control
![Histogram of Angle based action control](gym-results/2-Histogram_AngleBased_Action.png)

### 1.3 Random Search based control
![Plots of Random Search based control](gym-results/3-RandomSearchPlots.png)

### 1.4 Hill climbing
![Plots of Hill Climbing](gym-results/4-HillClimbingPlots.png)

### 1.5 Simulated Annealing
![Plots of Simulated Annealing](gym-results/5-SimulatedAnnealingPlots.png)

### 1.6 Adaptive noise scaling
![Plots of Adaptive Noise Scaling](gym-results/6-AdaptiveNoiseScalingPlots.png)
