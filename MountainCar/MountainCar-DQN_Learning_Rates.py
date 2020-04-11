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


class DQN:
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
        vector = self.model.predict_on_batch(states)
        index = np.array([i for i in range(self.batch_size)])
        vector[[index], [actions]] = tt

        #fit the model
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
            s = np.reshape(s, [1, self.num_observation_space])
            
			#loop for 500 steps (limit it to avoid hover)
            for step in range(250):
                #env.render()
                
				#choose the best action based on the greedy formula
                action = self.choose_action(s)
                
				#take the step based on the action
                new_s, reward, done, info = env.step(action)
                new_s = np.reshape(new_s, [1, self.num_observation_space])
                
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
            
			#break out of loop if we are already achieving above a 200
            current_mean_rewards = np.mean(self.rewards_list[-50:])
            if current_mean_rewards > 200:
                break
            
            #Decay Epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            print("The reward for episode " + str(episode) + " is " + str(episode_reward) + " with a mean average reward for the last 50 episodes of " + str(current_mean_rewards))
        
        #return the trained model
        return self.model, self.rewards_list

def test_model(trained_model, env):
    #initialize rewards vector
    rewards = []

    #test the model for 100 episodes
    for e in range(100):
        s = env.reset()
        s = np.reshape(s, [1, self.num_observation_space])
        
        #intialize episodic reward
        episode_reward = 0
        done = False
        while not done:
            env.render()
            predictions = trained_model.predict(s)
            action = np.argmax(predictions[0])
            new_s, reward, done, info = env.step(action)
            new_s = np.reshape(new_s, [1, self.num_observation_space])
            s = new_s
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        print("The reward for episode ", e, " is ", episode_reward)
    return rewards

def plot(df1, df2, df3, df4, chart_name, name, x, y):
	plt.rcParams.update({'font.size': 17})
	plt.figure(figsize=(15, 8))
	plt.close()
	plt.figure()

	plot = df1.plot(linewidth=0, figsize=(15, 8), title=name)
	exp1 = df1.rolling(window=100).mean()
	plt.plot(exp1, label = "LR=0.0001")

	exp2 = df2.rolling(window=100).mean()
	plt.plot(exp2, label = "LR=0.001")

	exp3 = df3.rolling(window=100).mean()
	plt.plot(exp3, label = "LR=0.01")

	exp4 = df4.rolling(window=100).mean()
	plt.plot(exp4, label = "LR=0.1")

	plot.set_xlabel(x)
	plot.set_ylabel(y)
	fig = plot.get_figure()
	plt.legend().set_visible(True)
	fig.savefig(chart_name)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    #Hyper Params
    gamma = .99
    epsilon = 1.0
    epsilon_decay = .97 #.996
    epsilon_min = .0 #.01 - .1
    memory_limit = 100000 #100000-500000
    net_replacement_freq = 5 #5-20
    batch_size = 64 #64-256
    max_episodes = 4000 #1000-2000

    den1 = 512 #150
    den2 = 256 #120
    env.seed(10)
    np.random.seed(100)
    
    lr = 0.0001
    #Build the model
    model = DQN(env, memory_limit, lr, den1, den2, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, net_replacement_freq)
    #train the model
    trained_model, rewards_list1 = model.train(max_episodes)
    
    lr = 0.001
    #Build the model
    model = DQN(env, memory_limit, lr, den1, den2, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, net_replacement_freq)
    #train the model
    trained_model, rewards_list2 = model.train(max_episodes)
    
    lr = 0.01
    #Build the model
    model = DQN(env, memory_limit, lr, den1, den2, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, net_replacement_freq)
    #train the model
    trained_model, rewards_list3 = model.train(max_episodes)
    lr = 0.1
    #Build the model
    model = DQN(env, memory_limit, lr, den1, den2, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, net_replacement_freq)
    #train the model
    trained_model, rewards_list4 = model.train(max_episodes)
    
    #print rewards for each training episode
    plot(pd.DataFrame(rewards_list1), pd.DataFrame(rewards_list2),pd.DataFrame(rewards_list3),pd.DataFrame(rewards_list4),"MountainCar-DQN Learning Rates", "MountainCar-DQN Learning Rates", "Episode", "Reward")
    


