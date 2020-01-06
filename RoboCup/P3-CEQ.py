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
	plt.title("Correlated-Q")
	ax1.plot(Q_updates, linewidth=0.1)
	plt.show()


###################################

def correlated_equilibrium(Q, op_Q, solver=None):

	Q = Q.flatten()
	op_Q = op_Q.flatten()

	solvers.options['show_progress'] = False
	solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}

	#c matrix
	c = -np.array(Q + op_Q, dtype="float")
	c = matrix(c)

	#Inequality constraints G*x <= h
	G = np.empty((0, 25))

	#Player constraints
	for x in range(5):
		for y in range(5):  
			if x == y: continue
			cons = np.zeros(25)
			base = x * 5
			comp = y * 5
			for _ in range(5):
				cons[base + _] = Q[comp + _] - Q[base + _]
			G = np.vstack([G, cons])

	#Opponent constraints
	op_G = np.empty((0, 25))
	for x in range(5): 
		for y in range(5): 
			if x == y: continue
			cons = np.zeros(25)
			for _ in range(5):
				cons[x + _ * 5] = op_Q[y + (_ * 5)] - op_Q[x + (_ * 5)]
			op_G = np.vstack([op_G, cons])

	G = np.vstack([G, op_G])
	G = np.matrix(G, dtype="float")
	G = np.vstack([G, -1. * np.eye(25)])
	h_size = len(G)
	G = matrix(G)
	
	#h matrix
	h = matrix(np.array(np.zeros(h_size), dtype="float"))

	#A matrix
	A = matrix(np.matrix(np.ones(25), dtype="float"))
	
	#b matrix
	b = matrix(np.matrix(1, dtype="float"))

	sol = solvers.lp(c, G, h, A, b, solver=solver)

	#probability
	probs = np.array(sol['x'].T)[0]
	probs -= probs.min() + 0.
	return probs.reshape((5, 5)) / probs.sum(0)

#############################

class Correlated_Q():
	
	#Based on Greenwald (2003) // Section 3.1, Equation (9), page 3
	

	def __init__(self):
		self.ns = 128
		self.na = 5
		
		# Q table
		self.Q = np.random.randn(self.ns, self.na, self.na)
		# self.Q = np.ones((self.ns, self.na, self.na))
		self.V = np.ones(self.ns)
		self.pi = np.zeros((self.ns, self.na))
		self.pi.fill(1. / self.na)

		# Learning rate
		self.alpha = 1.0
		self.alpha_decay = 0.999997
		self.alpha_min = 0.0

		# Random action rate
		self.epsilon = 0.75
		self.epsilon_decay = 0.9995
		self.epsilon_min = 0.01

		# Discount rate
		self.gamma = 0.9
		
	def select_first_action(self, state):
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
			self.epsilon *= self.epsilon_decay
			#self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  #epsilon decay
		else:
			try:
				action = np.random.choice(5, p=self.pi[new_state])
			except ValueError:
				#sum to 1
				self.pi[state] = self.pi[new_state] / self.pi[new_state].sum(0)
				action = np.random.choice(self.na, p=self.pi[new_state])


		return action
		
	def update_Q(self, state, action, opponent_action, new_state, reward, op_Q):
		#calculate previous Q
		old_Q = self.Q[state, action, opponent_action]

		# Update pi[s]
		self.pi[state] = np.sum(np.array(correlated_equilibrium(self.Q[new_state], op_Q[new_state])).reshape((5, 5)), axis=1)

		#Update V[new_state]
		self.V[new_state] = sum([self.pi[new_state, action_] * self.Q[new_state, action_, opponent_action] for action_ in range(5)])

		#Update Q[s,a,o]
		new_Q = (1 - self.alpha) * old_Q + self.alpha * (reward + self.gamma * self.V[new_state])
		self.Q[state, action, opponent_action] = new_Q

		#Decay alpha
		self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
		
		#return Delta Q
		return abs(new_Q - old_Q)

##################################

env = SoccerEnv()
Q_updates = []
A_agent = Correlated_Q()
B_agent = Correlated_Q()

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
	B_action = np.random.choice(5)
	#B_action = B_agent.select_first_action(state)
	A_action = A_agent.select_first_action(state)

	while t < game_length: #num steps per game
		t += 1
		
		#Execute joint action and observe new state and rewards
		new_state, A_reward, done, details = env.step(env.encode_action(A_action, B_action))
		B_reward = -A_reward

		#for all agents k, Update Q Table
		#B_Q_Update = B_agent.update_Q(state, B_action, A_action, new_state, B_reward, A_agent.Q)
		#A_Q_Update = A_agent.update_Q(state, A_action, B_action, new_state, A_reward, B_agent.Q)
		A_Q_Update = A_agent.update_Q(state, A_action, B_action, new_state, A_reward, 1 - A_agent.Q)

		#select next agent action
		B_action = np.random.choice(5)
		#B_action = B_agent.select_action(state, new_state)
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
#print (Q_updates)
plot(Q_updates)