import matplotlib.pyplot as plt
import numpy as np
from emoudahi_env import SoccerEnv
np.random.seed(100)

###############################

def plot(Q_updates):
	fig, ax1 = plt.subplots()
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
	ax1.set_xlabel('Simulation Iteration')
	ax1.set_ylabel('Q-value Difference')
	ax1.set_ylim((0, 0.5))
	ax1.set_xlim((0, 1000000))
	ax1.tick_params('y')
	fig.tight_layout()
	plt.title("Friend-Q")
	ax1.plot(Q_updates, linewidth=0.1)
	plt.show()


###################################

class Friend_Q():
	
	#See Littman (2001) Equation (6), page 2 // Equation (7), page 4

	def __init__(self):
		# Q/V table
		self.Q = np.random.randn(128, 5, 5)
		self.V = np.ones(128)

		# Learning rate
		self.alpha = 1.0
		self.alpha_decay = 0.9999 #(1/n(s,a) where n(s,a) is number of times at state
		self.alpha_min = 0.0

		# Random action rate
		self.epsilon = 0.75
		self.epsilon_decay = 0.99999 #.99995
		self.epsilon_min = 0.01

		# Discount rate
		self.gamma = 0.9


	def select_first_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice(5) #choose random
			self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  #epsilon decay
		
		else:
			action, op_action = np.unravel_index(np.argmax([self.Q[state]]), self.Q[state].shape)

		return action
		
	def select_action(self, state, new_state):
		if np.random.random() < self.epsilon:
			action = np.random.choice(5) #choose random
			self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  #epsilon decay
		
		else:
			#action, op_action = np.unravel_index(np.argmax([self.Q[state]]), self.Q[state].shape)
			nash = np.argmax([self.Q[new_state]], axis=None)
			action, op_action = np.unravel_index(nash, self.Q[state].shape)

		return action
		

	def update_Q(self, state, action, opponent_action, new_state, reward):
	
		#set previous Q
		prev_Q = self.Q[state, action, opponent_action]

		# Calculate Nash_i(s, Q_1, Q_2)
		nash = np.max(self.Q[state])
		
		#calculated new Q value
		updated_Q = (1 - self.alpha) * prev_Q + self.alpha * 
					(reward + self.gamma * self.V[new_state])

		#Update Q table
		self.Q[state, action, opponent_action] = updated_Q

		# Update V[s] with Nash_i(s, Q_1, Q_2)
		self.V[state] = nash

		#alpha decay
		self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
		
		#Return Delta Q
		return abs(updated_Q - prev_Q)
		
######################################		
		
		
		
env = SoccerEnv()
Q_updates = []

B_agent = Friend_Q()
A_agent = Friend_Q()
max_steps = 1000000
t = 0

while t <= max_steps:
	t += 1

	#loading bar
	if t % 1000 == 0:
		print(t / max_steps)

	#Reset env
	state = env.reset()
	
	#fixed starting state?
	env.s = env.encode_state(0, 2, 0, 1, 0)

	#initialize variables for game
	done = False
	game_length = t + 500

	#select first agent action
	B_action = B_agent.select_first_action(state)
	A_action = A_agent.select_first_action(state)

	while t < game_length: #num steps per game
		t += 1
		


		#Execute joint action and observe new state and rewards
		new_state, A_reward, done, details = env.step(env.encode_action(A_action, B_action))
		B_reward = -A_reward

		#for all agents k, Update Q Table
		B_Q_Update = B_agent.update_Q(state, B_action, A_action, new_state, B_reward)
		A_Q_Update = B_agent.update_Q(state, A_action, B_action, new_state, A_reward)

		#select next agent action
		B_action = B_agent.select_action(state, new_state)
		A_action = A_agent.select_action(state, new_state)

		#5a. s = s'
		state = new_state	
		
		if state == 17 and B_action == 4 and A_action == 1:
			Q_updates.append(A_Q_Update)
		elif len(Q_updates) > 0:
			Q_updates.append(Q_updates[-1])


		if done:
			#env.render()
			break


################
plot(Q_updates)