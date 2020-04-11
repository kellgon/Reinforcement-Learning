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

#env = gym.make('FrozenLake8x8-v0')
env = gym.make('FrozenLake-v0')
env = env.unwrapped
env.render()

# V and params
V = np.zeros((env.observation_space.n))
pi = np.ones([env.nS, env.nA]) / env.nA
discount = 0.99
last_delta = 10
rounds = 0
deltas = []

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

def one_step_lookahead(state, V):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount * V[next_state])
    return A


def policy_eval(pi, env, discount=0.9, theta=0.0001):


	V = np.zeros((env.observation_space.n))
	while True:
		delta = 0
		# For each state, perform a "full backup"
		for s in range(env.nS):
			v = 0
			# Look at the possible next actions
			for a, action_prob in enumerate(pi[s]):
			    # For each action, look at the possible next states...
			    for  prob, next_state, reward, done in env.P[s][a]:
			        # Calculate the expected value
			        v += action_prob * prob * (reward + discount * V[next_state])
			# How much our value function changed (across any states)
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		# Stop evaluating once our value function change is below a threshold
		deltas.append(delta)
		#print(delta)
		if delta < theta:
			break
	return np.array(V)

#Policy Improvement   
while True:
	V = policy_eval(pi, env, discount)
	# Will be set to false if we make any changes to the policy
	policy_stable = True

	# For each state...
	for s in range(env.nS):
		# The best action we would take under the current policy
		a = np.argmax(pi[s])
		
		# Find the best action by one-step lookahead
		# Ties are resolved arbitarily
		A = one_step_lookahead(s, V)
	 
		best_action = np.argmax(A)
		
		# Greedily update the policy
		if a != best_action:
		    policy_stable = False
		pi[s] = np.eye(env.nA)[best_action]

	# If the policy is stable we've found an optimal policy. Return it
	if policy_stable:
		break




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
#plot_rewards(pd.DataFrame(rewards), "Frozen Lake Policy-Iteration Performance", "Frozen Lake Policy-Iteration Performance", "Episode", "Rolling Average Rewards")
plot_converge(pd.DataFrame(deltas), "Frozen Lake Policy-Iteration Convergence", "Frozen Lake Policy-Iteration Convergence", "Iteration", "Delta")
print_policy(pi)
#print(pi)
#print(V)
