'''
The code is partially referred from 1. https://github.com/kelvin84hk/DRLND_P3_collab-compet

'''
import numpy as np
from src.model import Actor, CriticD4PG
import torch.optim as optim
import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class d4pg():

	def __init__(self,
				 state_size,
				 action_size,
				 seed,
				 af,
				 lr_actor,
				 lr_critic,
				 n_atoms,
				 vmax,
				 vmin,
				 epsilon=0.3,
				 device=device):
		self.af= af
		self.state_size = state_size
		self.action_size = action_size
		self.device = device
		self.epsilon = epsilon
		# Actor Network
		self.actor_local = Actor(state_size, action_size, seed, fc1_units=64, fc2_units=64,
								 mode=self.af).to(device)
		self.actor_target = Actor(state_size, action_size, seed, fc1_units=64, fc2_units=64,
								  mode=self.af).to(device)
		# Optimizer Actor
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
		# Critic Network
		self.critic_local = CriticD4PG(state_size, action_size, seed, n_atoms, vmax, vmin, fc1_units=64, fc2_units=64,
									   mode=self.af).to(device)
		self.critic_target = CriticD4PG(state_size, action_size, seed, n_atoms, vmax, vmin, fc1_units=64, fc2_units=64,
										mode=self.af).to(device)
		# Optimizer Critic
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)

		# Initialize Networks
		self.hard_update(self.actor_local, self.actor_target)
		self.hard_update(self.critic_local, self.critic_target)



	def hard_update(self, local_model, target_model):
		"""
		copy parameters from local to target
		:param local_model:
		:type local_model:
		:param target_model:
		:type target_model:
		:return:
		:rtype:
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(local_param.data)

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		======
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

	def act(self, state, mode):
		"""

		:param state (1, state_size) (numpy): state for a single agent
		:param mode (string): 'test' or 'train' mode
		:return: action (1, action_size): action for a single agent
		"""
		state_v = torch.Tensor(np.array(state, dtype=np.float32)).to(self.device)

		self.actor_local.eval()
		with torch.no_grad():
			mu_v = self.actor_local(state_v)
			action = mu_v.data.cpu().numpy()
		self.actor_local.train()

		if mode == "test":
			return np.clip(action, -1, 1)

		elif mode == "train":
			action += self.epsilon * np.random.normal(size=action.shape)
			action = np.clip(action, -1, 1)
			return action


