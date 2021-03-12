from memory import Transition, ReplayMemory
from model import DQN

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Agent:
	def __init__(self, state_size=14, T=96, is_eval=True):
		self.state_size = state_size # normalized previous days
		self.action_size = 3
		self.memory = ReplayMemory(10000)
		self.inventory = []
		self.is_eval = is_eval
		self.T = T

		self.gamma = 0.99
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.batch_size = 16
		if os.path.exists('models/target_model'):
			self.policy_net = torch.load('models/policy_model', map_location=device)
			self.target_net = torch.load('models/target_model', map_location=device)
		else:
			self.policy_net = DQN(state_size, self.action_size).to(device)
			self.target_net = DQN(state_size, self.action_size).to(device)

			for param_p in self.policy_net.parameters(): 
				weight_init.normal_(param_p)

		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.00025)
		
	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size) - 1

		tensor = torch.FloatTensor(state).to(device)
		tensor = tensor.unsqueeze(0)
		options = self.target_net(tensor)
		# options = self.policy_net(tensor)
		return (np.argmax(options[-1].detach().cpu().numpy()) - 1)
		# return (np.argmax(options[0].detach().numpy()) - 1)

	def store(self, state, actions, new_states, rewards, action, step):
		if step < 1000: # soft update
			for n in range(len(actions)):
				self.memory.push(state, actions[n], new_states[n], rewards[n])
		else:
			for n in range(len(actions)):
				if actions[n] == action:
					self.memory.push(state, actions[n], new_states[n], rewards[n])
					break

	def optimize(self, step):
		# print(len(self.memory))
		if len(self.memory) < self.batch_size * 10:
				return
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state = torch.FloatTensor(batch.next_state).to(device)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
		non_final_next_states = torch.cat([s for s in next_state if s is not None])

		state_batch = torch.FloatTensor(batch.state).to(device)
		action_batch = torch.LongTensor(torch.add(torch.tensor(batch.action), torch.tensor(1))).to(device)
		reward_batch = torch.FloatTensor(batch.reward).to(device)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		l = self.policy_net(state_batch).size(0)
		state_action_values = self.policy_net(state_batch)[95:l:96].gather(1, action_batch.reshape((self.batch_size, 1)))
		state_action_values = state_action_values.squeeze(-1)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.batch_size, device=device)
		next_state_values[non_final_mask] = self.target_net(next_state)[95:l:96].max(1)[0].detach()
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch

		# Compute the loss
		loss = torch.nn.MSELoss()(expected_state_action_values, state_action_values)

		# Optimize the model
		
		loss.backward()
		for param in self.policy_net.parameters():
				param.grad.data.clamp_(-1, 1)
		
		self.optimizer.step()
		
		if step % self.T == 0:
			# print('soft_update')
			gamma = 0.001
			param_before = copy.deepcopy(self.target_net)
			target_update = copy.deepcopy(self.target_net.state_dict())
			for k in target_update.keys():
				target_update[k] = self.target_net.state_dict()[k] * (1 - gamma) + self.policy_net.state_dict()[k] * gamma
			self.target_net.load_state_dict(target_update)

