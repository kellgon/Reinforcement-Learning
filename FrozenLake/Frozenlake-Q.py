import gym
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot(df, chart_name, name, x, y):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=0, figsize=(15, 8), title=name)
    exp1 = df.rolling(window=100).mean()
    plt.plot(exp1)
    plot.set_xlabel(x)
    plot.set_ylabel(y)
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)

#env = gym.make('FrozenLake8x8-v0')
env = gym.make('FrozenLake-v0')
env = env.unwrapped

# Q and rewards
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
iterations = []
updates = []

# Parameters
alpha = 0.75
discount = 0.9
episodes = 2000

# Episodes
for episode in range(episodes):
	# Refresh state
	state = env.reset()
	done = False
	t_reward = 0
	max_steps = 200

	# Run episode
	for i in range(max_steps):
		if done:
			break

		current = state
		action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

		state, reward, done, info = env.step(action)
		t_reward += reward
	
		
		Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

		#env.render()
	rewards.append(t_reward)
	

# Close environment
env.close()

# Plot results

#print rewards for each training episode
plot(pd.DataFrame(rewards), "Frozen Lake Q-Learner2", "Frozen Lake Q-Learner2", "Episode", "Rolling Average Rewards")
print(rewards)
