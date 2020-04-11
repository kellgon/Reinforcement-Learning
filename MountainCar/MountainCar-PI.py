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

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# V and params
num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

# V and params
V = np.zeros((num_states))
pi = np.ones([19, 15, env.action_space.n])
discount = 0.9
last_delta = 10
rounds = 0
deltas = []
not_converged = True


def one_step_lookahead(state, V):
	A = np.zeros(env.action_space.n)
	for a in range(env.action_space.n):
		temp = env
		new_state, reward, done, info = temp.step(a) 
		new_state_adj = (new_state - env.observation_space.low)*np.array([10, 100])
		new_state_adj = np.round(new_state_adj, 0).astype(int)
		A[a] += (reward + discount * V[new_state_adj[0], new_state_adj[1]])
	return A


def policy_eval(pi, env, discount=0.9, theta=0.0001):

	V = np.zeros((num_states))
	while True:
		delta = 0
		# For each state, perform a "full backup"
		for x in range(19):
			for y in range(15):
				v = 0
				# Look at the possible next actions
				#for a, action_prob in enumerate(pi[s]):
					# For each action, look at the possible next states..
					#for  prob, next_state, reward, done in env.P[s][a]:
					#    # Calculate the expected value
					#    v += action_prob * prob * (reward + discount * V[next_state])
				for a in range(env.action_space.n):
					temp = env
					new_state, reward, done, info = temp.step(a) 
					new_state_adj = (new_state - env.observation_space.low)*np.array([10, 100])
					new_state_adj = np.round(new_state_adj, 0).astype(int)
					v += (1/env.action_space.n)*(reward + discount * V[new_state_adj[0], new_state_adj[1]])


				delta = max(delta, np.abs(v - V[x,y]))
				V[x,y] = v
				# Stop evaluating once our value function change is below a threshold
		deltas.append(delta)
		print(delta)
		if delta < theta:
			print("yay")
			break
	return np.array(V)

#Policy Improvement   
while True:
	V = policy_eval(pi, env, discount)
	break
	# Will be set to false if we make any changes to the policy
	policy_stable = True
	print(V)
	# For each state...
	for x in range(19):
		for y in range(15):
			s = [x,y]
			# The best action we would take under the current policy
			a = np.argmax(pi[s])
			
			# Find the best action by one-step lookahead
			# Ties are resolved arbitarily
			A = one_step_lookahead(s, V)
		 
			best_action = np.argmax(A)
			
			# Greedily update the policy
			if a != best_action:
				policy_stable = False
			pi[s] = np.eye(env.action_space.n)[best_action]

		# If the policy is stable we've found an optimal policy. Return it
		if policy_stable:
			break




# Graph the V Learner
rewards = []
episodes = 50
for episode in range(episodes):
	# Refresh state
	state = env.reset()
	state_adj = (state - env.observation_space.low)*np.array([10, 100])
	state_adj = np.round(state_adj, 0).astype(int)
	done = False
	t_reward = 0
	max_steps = 2000

	# Run episode
	for i in range(max_steps):
		if done:
			break
		#env.render()
		action = np.argmax(pi[state_adj, :])
		state, reward, done, info = env.step(action)
		t_reward += reward
		state_adj = (state - env.observation_space.low)*np.array([10, 100])
		state_adj = np.round(state_adj, 0).astype(int)
	
		#env.render()
	rewards.append(t_reward)
	print(episode, t_reward)
	
# Close environment
env.close()


# Plot results

#print rewards for each training episode
plot_rewards(pd.DataFrame(rewards), "Mountain Car Policy-Iteration Performance", "Mountain Car Policy-Iteration Performance", "Episode", "Rolling Average Rewards")
plot_converge(pd.DataFrame(deltas), "Mountain Car Policy-Iteration Convergence", "Mountain Car Policy-Iteration Convergence", "Iteration", "Delta")
