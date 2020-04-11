import gym
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_rewards(df, chart_name, name, x, y):
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
    
def plot_converge(df, chart_name, name, x, y):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(15, 8), title=name)
    plot.set_xlabel(x)
    plot.set_ylabel(y)
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)
    
def print_policy(pi):
	for i in range(16):
		if i % 4 == 0:
			print("")
		
		A = pi[i]
		if A[0] == 1:
			print('<', end='')
		if A[1] == 1:
			print('v', end='')
		if A[2] == 1:
			print('>', end='')
		if A[3] == 1:
			print('^', end='')
		

#env = gym.make('FrozenLake8x8-v0')
env = gym.make('FrozenLake-v0')
env = env.unwrapped
env.render()

# V and params
V = np.zeros((env.observation_space.n))
discount = 0.99
last_delta = 10
rounds = 0
deltas = []

#Build the V table
while last_delta > .0001:
	rounds += 1
	delta = 0
	for state in range(env.observation_space.n):
		temp = V[state]
		A = np.zeros(env.action_space.n)
		for a in range(env.action_space.n):
			for prob, new_state, reward, done in env.P[state][a]:
				A[a] += prob * (reward + discount * V[new_state])

		V[state] = np.max(A)

		delta = max(delta, np.abs(temp - V[state]))


	last_delta = delta
	#print(rounds, last_delta)
	deltas.append(delta)
	
#Build the optimal policy off the optimal V table
pi = np.zeros((env.observation_space.n, env.action_space.n))
for state in range(env.observation_space.n):

	A = np.zeros(env.action_space.n)
	for a in range(env.action_space.n):
		for prob, new_state, reward, done in env.P[state][a]:
			A[a] += prob * (reward + discount * V[new_state])
	best_action = np.argmax(A)

	pi[state, best_action] = 1.0


# Graph the V Learner
rewards = []
episodes = 1000
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
		action = np.argmax(pi[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

		state, reward, done, info = env.step(action)
		t_reward += reward
	
		#env.render()
	rewards.append(t_reward)
	
# Close environment
env.close()

# Plot results

#print rewards for each training episode
plot_rewards(pd.DataFrame(rewards), "Frozen Lake Value-Iteration Performance", "Frozen Lake Value-Iteration Performance", "Episode", "Rolling Average Rewards")
#plot_converge(pd.DataFrame(deltas), "Frozen Lake Value-Iteration Convergence", "Frozen Lake Value-Iteration Convergence", "Iteration", "Delta")
env.reset()
print_policy(pi)
print(pi)
print(V)
