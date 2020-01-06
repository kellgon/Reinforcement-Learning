import sys
from contextlib import closing

import numpy as np
from six import StringIO, b
import pandas as pd

import gym
import gym.envs.toy_text;
import random

from gym import utils
from gym.envs.toy_text import discrete
from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.models import load_model
import pickle
from matplotlib import pyplot as plt


class Lunar:
    def __init__(self, env, memory_limit, lr, den1, den2, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, net_replacement_freq):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.net_replacement_freq = net_replacement_freq
        self.lr = lr
        self.rewards_list = []
        self.memory_limit = deque(maxlen=memory_limit) #Initialize replay memory D to capacity N
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        self.model = self.initialize_model(den1, den2)

        #Initialize the model - action-value function Q with random weights
    def initialize_model(self, den1, den2):
        model = Sequential()
        model.add(Dense(den1, input_dim=self.num_observation_space, activation='relu',  kernel_initializer=keras.initializers.RandomNormal(seed=10)))
        model.add(Dense(den2, activation='relu', kernel_initializer=keras.initializers.RandomNormal(seed=100)))
        model.add(Dense(self.num_action_space, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=1000)))
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        return model

    def updateQ(self, episode):

        #test to see if rewards are already high enough, the memory limit is full, or we are on the net replacement freq
        if len(self.memory_limit) < self.batch_size:
            return
        elif episode % self.net_replacement_freq == 0:
            return
        elif np.mean(self.rewards_list[-10:]) > 170: 
            return

        #use the minibatch to update
        minibatch = random.sample(self.memory_limit, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        new_states = np.array([i[3] for i in minibatch])
        done_list = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        new_states = np.squeeze(new_states)
        states = np.squeeze(states)
        
        #update TT
        tt = rewards + self.gamma * (np.amax(self.model.predict_on_batch(new_states), axis=1)) * (1 - done_list)
        
        #Get current Q values
        vector = self.model.predict_on_batch(states)
        
        #Update Q values
        index = np.array([i for i in range(self.batch_size)])
        vector[[index], [actions]] = tt

        #fit the model (gradient Descent)
        self.model.fit(states, vector, epochs=1, verbose=0)

    def choose_action(self, s):
        #if less random less than epsilon, choose a random action
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)
        else: #choose the best action
        	predictions = self.model.predict(s)
        	return np.argmax(predictions[0])


    def train(self, max_episodes):
        for episode in range(max_episodes):
            #reset state
            s = env.reset()
            
			#reset episode award
            episode_reward = 0
            s = np.reshape(s, [1, 8])
            
			#loop for x amount of steps (limit it to avoid hover)
            for step in range(250):
                #env.render()
                
				#choose the best action based on the greedy formula
                action = self.choose_action(s)
                
				#take the step based on the action
                new_s, reward, done, info = env.step(action)
                new_s = np.reshape(new_s, [1, 8])
                
				#record the outcome in the memory file
                self.memory_limit.append((s, action, reward, new_s, done))
                
				#s=s'
                s = new_s
                
				#update the episodic reward
                episode_reward += reward
                
                self.updateQ(episode)

				#stop looping one you are done
                if done:
                    break
                    
            #update the rewards list to keep track of previous episodes
            self.rewards_list.append(episode_reward)
            
            #Decay Epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            print("The reward for episode " + str(episode) + " is " + str(episode_reward) + " with a mean average reward for the last 50 episodes of " + str(current_mean_rewards))
        
        #return the trained model
        return self.model, self.rewards_list

def plot(df, chart_name, name, x, y):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(16, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(16, 8), title=name)
    plot.set_xlabel(x)
    plot.set_ylabel(y)
    plt.ylim((-600, 300))
    fig = plot.get_figure()
    fig.savefig(chart_name)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    #Hyper Params
    lr = .001
    gamma = .99
	epsilon = 1.0
    epsilon_decay = .995 #.996
    epsilon_min = .01 #.01 - .1
    #memory_limit = 500000 #100000-500000
    net_replacement_freq = 5 #5-20
    batch_size = 64 #64-256
    max_episodes = 1000 #1000-2000
    den1 = 512 #150 -512
    den2 = 256 #120 - 256
    env.seed(10)
    np.random.seed(100)
    pd1 = pd.DataFrame(index=pd.Series(range(1, max_episodes+1)))
   
    #run the experiments
    for i in range(0,4):
        if i == 0:
            memory_limit = 50000
        elif i == 1:
            memory_limit = 100000
        elif i == 2:
            memory_limit = 250000
        elif i == 3:
            memory_limit = 500000
            
        #Build the models
        model = Lunar(env, memory_limit, lr, den1, den2, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, net_replacement_freq)
		
		#train the model
        trained_model, rewards_list = model.train(max_episodes)
	
        name = "Memory Limit = " + str(lr)
        pd1[name] = rewards_list
		#print rewards for each testing episode
    plot(pd1, "Memory test", "Testing Memory Rates", "Episode","Reward")

