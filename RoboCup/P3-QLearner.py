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
	plt.title("QLearner")
	ax1.plot(Q_updates, linewidth=0.1)
	plt.show()


###################################

class QLearner():
	#Greenwald (2008) Table 1, page 3
	
	def __init__(self):
		# Q table
		self.Q = np.random.randn(128, 5)

		# Learning rate
		self.alpha = 0.5
		self.alpha_decay = 0.999997
		self.alpha_min = 0.0

		# Random action rate
		self.epsilon = 0.75
		self.epsilon_decay = 0.99995
		self.epsilon_min = 0.01

		# Discount rate
		self.gamma = 0.9


	def select_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice(5)
			self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
		else:
			action = np.argmax(self.Q[state])

		return action

	def update_Q(self, state, action, new_state, reward):
		
		new_action = np.argmax(self.Q[state])
		
		#track the previous Q value
		old_Q = self.Q[state, action]
		
		#track the updated Q value
		new_Q = (1 - self.alpha) * old_Q + self.alpha * ((1 - self.gamma) * 
		         reward + self.gamma * self.Q[new_state, new_action] - old_Q)
		
		
		#update the Q table
		self.Q[state, action] = new_Q
		
		#alpha decay
		self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
		
		#return delta Q
		return abs(new_Q - old_Q)
		
		
######################################

env = SoccerEnv()
Q_updates = []

B_agent = QLearner()
A_agent = QLearner()
max_steps = 1000000

for t in range(max_steps): #run for exactly 10 x 10^5 steps

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
	while t < game_length: #num steps per game
		t += 1
		
		#Choose actions
		#B_action = env.decode_action(B_agent.select_action(state))[0]
		#A_action = env.decode_action(A_agent.select_action(state))[0]
		
		B_action = B_agent.select_action(state)
		A_action = A_agent.select_action(state)
		
		#Execute joint action and observe new state and rewards
		new_state, A_reward, done, details = env.step(env.encode_action(A_action, B_action))
		B_reward = -A_reward
		
		#Update Q Table
		B_Q_Update = B_agent.update_Q(state, B_action, new_state, B_reward)
		A_Q_Update = B_agent.update_Q(state, A_action, new_state, A_reward)

		#5a. s = s'
		state = new_state	
		
		#Add to Q update table
		if state == 17 and A_action == 1:
			Q_updates.append(A_Q_Update)
		elif len(Q_updates) > 0:
			Q_updates.append(Q_updates[-1])
		

		if done:
			break


################
plot(Q_updates)