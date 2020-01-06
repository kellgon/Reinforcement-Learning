import matplotlib.pyplot as plt
import numpy as np
from emoudahi_env import SoccerEnv
from cvxopt import matrix, solvers
np.random.seed(100)

###############################

def plot(Q_updates):
	fig, ax1 = plt.subplots()
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
	ax1.set_xlabel('Simulation Iteration')
	ax1.set_ylabel('Q-value Difference')
	ax1.set_ylim((0, 0.5))
	#ax1.set_xlim((0, 1000000))
	ax1.tick_params('y')
	fig.tight_layout()
	plt.title("Foe-Q")
	ax1.plot(Q_updates, linewidth=0.1)
	plt.show()


###################################

def minimax(Q):
	solvers.options['show_progress'] = False
	solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}

	c = matrix([-1., 0., 0., 0., 0., 0.])

	# Inequality constraints
	G = -1 * (np.matrix(Q).T) 
	G = np.vstack([G, np.eye(5)*-1])  
	utility_column =np.append(np.ones(5), np.zeros(5))
	G =np.insert(G, 0, utility_column, axis=1)
	G = matrix(G)

	h= matrix([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
	#h= matrix([0., 0., 0., 0., 0., 0.])

	A = np.matrix([0., 1., 1., 1., 1., 1.])
	A = matrix(A)

	b = matrix([1.])

	solve = solvers.lp(c, G, h, A, b, solver='glpk')

	probs = np.array(solve['x'].H[1:].T)[0]
	

	# Scale and normalize to prevent negative probabilities
	probs -= probs.min() + 0.

	return probs / probs.sum(0)

class Foe_Q():
	
	def __init__(self):
		# Q table
		#let Q[s,a,p] = 1
		self.Q = np.ones((128, 5, 5))
		
		#let V[s] = 1
		self.V = np.zeros(128)
		self.pi = np.empty((128, 5))
		#Let pi[s,a] = 1/A
		self.pi.fill(1.0 / 5)

		# Learning rate
		self.alpha = 1.0
		self.alpha_decay = 0.999997 #0.999
		self.alpha_min = 0.0 #0.001 #0.01

		# Random action rate
		self.epsilon = 0.2 # 0.75 #
		self.epsilon_decay = 0.9999954 # 0.9995 #
		self.epsilon_min = 0.01 

		# Discount rate
		self.gamma = 0.9
		
		
	def select_first_action(self, state):
		#print (self.pi)
		if np.random.random() < self.epsilon:
			action = np.random.choice(5)
			self.epsilon *= self.epsilon_decay
			#self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  #epsilon decay
		else:
			#Return action a with probability pi[s,a]
			try:
				action = np.random.choice(range(5), p=self.pi[state])
			except ValueError:
				#probabilities have to sum to 1
				self.pi[state] = self.pi[state] / self.pi[state].sum(0)
				action = np.random.choice(range(5), p=self.pi[state])

		return action

	def select_action(self, state, new_state):

		if np.random.random() < self.epsilon:
			action = np.random.choice(5)
			self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  #epsilon decay
		else:
            #Return action a with probability pi[s,a]
			try:
				action = np.random.choice(range(5), p=self.pi[new_state])
			except ValueError:
				#probabilities have to sum to 1
				self.pi[state] = self.pi[state] / self.pi[state].sum(0)
				action = np.random.choice(range(5), p=self.pi[new_state])

		return action

	
	def update_Q(self, state, action, opponent_action, new_state, reward):
		#calculate previous Q
		old_Q = self.Q[state, action, opponent_action]
			
		#calculate updated Q
		new_Q = (1 - self.alpha) * old_Q + self.alpha * ((1 - self.gamma) * reward + self.gamma * self.V[new_state])
		
		#update Q table
		self.Q[state, action, opponent_action] = new_Q
		
		# Update pi
		self.pi[new_state] = minimax(self.Q[new_state])

		self.V[state] = sum([self.pi[state, a] * self.Q[state, a, opponent_action] for a in range(5)])

		# Decay alpha
		self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
		
		#Return Delta Q
		return abs(new_Q - old_Q)
			
##################################

	
env = SoccerEnv()
Q_updates = []
A_agent = Foe_Q()

max_steps = 2000000
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
	B_action = np.random.choice(5)
	A_action = A_agent.select_first_action(state)

	while t < game_length: #num steps per game
		t += 1
		
		#Execute joint action and observe new state and rewards
		new_state, A_reward, done, details = env.step(env.encode_action(A_action, B_action))
		B_reward = -A_reward

		#for all agents k, Update Q Table
		A_Q_Update = A_agent.update_Q(state, A_action, B_action, new_state, A_reward)

		#select next agent action
		B_action = np.random.choice(5)
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
