import numpy as np
import gym
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

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, gamma, epsilon, epsilon_min, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, size = (num_states[0], num_states[1], env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        episode_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
    
        while done != True:   
               
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            new_state, reward, done, info = env.step(action) 
            
            # Discretize new_state
            new_state_adj = (new_state - env.observation_space.low)*np.array([10, 100])
            new_state_adj = np.round(new_state_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and new_state[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + gamma*np.max(Q[new_state_adj[0], new_state_adj[1]]) - Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            # Update variables
            episode_reward += reward
            state_adj = new_state_adj
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay
        
        # Track rewards
        reward_list.append(episode_reward)
 
    env.close()
    
    return reward_list


if __name__ == '__main__':
	learning = 0.2 #.001
	gamma = 0.9 #.99

	min_eps = 0 #0.01
	episodes = 5000
	epsilon = 1.0
	epsilon_decay = .97 #.996
	epsilon_min = .01 #.01 - .1
	# Run Q-learning algorithm
	rewards = QLearning(env,learning, gamma, epsilon, min_eps, episodes)

	#print rewards for each training episode
	plot(pd.DataFrame(rewards), "Mountain Car Q-Learner", "Mountain Car Q-Learner", "Episode", "Rolling Average Rewards") 
