import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt


def plot(df, chart_name, name, x, y):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(15, 8), title=name)
    #exp1 = df.rolling(window=100).mean()
    #plt.plot(exp1)
    plot.set_xlabel(x)
    plot.set_ylabel(y)
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)

def print_policy(pi):
	for x in range(19):
		for y in range(15):
			A = pi[x,y]
			if A[0] == 1:
				print('<', end='')
			if A[1] == 1:
				print('v', end='')
			if A[2] == 1:
				print('-', end='')
		print("")

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# V and params
num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

V = np.zeros(num_states)
discount = 0.9
last_delta = 10
rounds = 0
deltas = []

while last_delta > .0001:
	rounds += 1
	delta = 0
	# Update each state...
	for x in range(19):
		for y in range(15):
			state = V[x,y]
			A = np.zeros(env.action_space.n)
			for a in range(env.action_space.n):
				temp = env
				new_state, reward, done, info = temp.step(a) 
				new_state_adj = (new_state - env.observation_space.low)*np.array([10, 100])
				new_state_adj = np.round(new_state_adj, 0).astype(int)

				A[a] += (reward + discount * V[new_state_adj[0], new_state_adj[1]])
			#print(A)
			V[x,y] = np.max(A)
			print(env, x,y,state, V[x,y])
			delta = max(delta, np.abs(state - V[x,y]))
	

	last_delta = delta
	print(rounds, last_delta)	
	deltas.append(last_delta)
	
#Build the optimal policy off the optimal V table
pi = np.zeros((19,15, env.action_space.n))
for x in range(19):
	for y in range(15):

		A = np.zeros(env.action_space.n)
		for a in range(env.action_space.n):
			temp = env
			new_state, reward, done, info = temp.step(a) 
			new_state_adj = (new_state - env.observation_space.low)*np.array([10, 100])
			new_state_adj = np.round(new_state_adj, 0).astype(int)
			A[a] += (reward + discount * V[new_state_adj[0], new_state_adj[1]])
		
		
		best_action = np.argmax(A)

		pi[x,y, best_action] = 1.0




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
#plot(pd.DataFrame(deltas), "Mountain Car Value-Iteration Convergence", "Mountain Car Value-Iteration Convergence", "Iteration", "Delta")
print_policy(pi)
