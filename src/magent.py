'''
The code is partially referred from 1. https://github.com/kelvin84hk/DRLND_P3_collab-compet

'''


from src.utils import ReplayBuffer
from src.agent import d4pg
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class multi_agent_d4pg():

	def __init__(self,
				 state_size,
				 action_size,
				 seed,
				 LR_ACTOR,
				 LR_CRITIC,
				 GAMMA,
				 REWARD_STEPS,
				 BUFFER_SIZE,
				 BATCH_SIZE,
				 N_ATOMS,
				 Vmax,
				 Vmin,
				 TAU,
				 LEARN_EVERY_STEP,
				 LEARN_REPEAT,
				 number_agents = 2,
				 af= 1

	):

		self.seed = seed
		self.state_size = state_size
		self.action_size = action_size
		self.t_step = 0
		self.gamma = GAMMA
		self.lr_actor = LR_ACTOR
		self.lr_critic = LR_CRITIC
		self.reward_steps = REWARD_STEPS
		self.buffer_size = BUFFER_SIZE
		self.batch_size = BATCH_SIZE
		self.n_atoms = N_ATOMS
		self.vmax = Vmax
		self.vmin = Vmin
		self.delta_z = (self.vmax - self.vmin) / (self.n_atoms - 1)
		self.tau = TAU
		self.learn_every_step = LEARN_EVERY_STEP
		self.learn_repeat = LEARN_REPEAT
		self.number_agents = number_agents
		self.af = af #
		# memory buffer
		self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
		# build an agent
		self.sagent = d4pg(self.state_size, self.action_size, self.seed, self.af, self.lr_actor,
									  self.lr_critic, self.n_atoms, self.vmax, self.vmin)
		if int(number_agents) > 1:

			self.mad4pg_agent = [ self.sagent for x in range(int(number_agents))]

		# placeholder to track loss
		self.critic_loss = []
		self.actor_loss = []

	def acts(self, states, mode):
		"""
		:param states (n_agents, state_size) (numpy): states for n agents
		:param mode (string): 'test' or 'train'
		:return: acts (n_agents, action_size) (numpy)
		"""
		acts = []
		for s, a in zip(states, self.mad4pg_agent):
			if len(s.shape) < 2:
				s = np.expand_dims(s, axis=0)
			acts.append(a.act(s, mode))
		return np.vstack(acts)

	def learn(self, agent, states, actions, rewards, next_states, dones, gamma):
		"""
		:param agent: one D4PG network
		:param
			states (batch_size, state_size) (tensor)
			actions (batch_size, action_size) (tensor)
			rewards (batch_size,) (tensor)
			next_states (batch_size, state_size) (tensor)
			dones (batch_size,) (tensor)
		:param gamma (float): discount factor
		"""
		# ---------------------------- update critic ---------------------------- #
		Q_expected = agent.critic_local(states, actions)
		actions_next = agent.actor_target(next_states)
		Q_targets_next = agent.critic_target(next_states, actions_next)

		Q_targets_next = F.softmax(Q_targets_next, dim=1)

		proj_distr_v = self.distr_projection(Q_targets_next, rewards, dones,
										gamma=gamma ** self.reward_steps, device=device)
		prob_dist_v = -F.log_softmax(Q_expected, dim=1) * proj_distr_v
		critic_loss_v = prob_dist_v.sum(dim=1).mean()

		# Minimize the loss
		agent.critic_optimizer.zero_grad()
		critic_loss_v.backward()
		torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
		agent.critic_optimizer.step()
		self.critic_loss.append(critic_loss_v.item())

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		actions_pred = agent.actor_local(states)
		crt_distr_v = agent.critic_local(states, actions_pred)
		actor_loss_v = -agent.critic_local.distr_to_q(crt_distr_v)
		actor_loss_v = actor_loss_v.mean()

		# Minimize the loss
		agent.actor_optimizer.zero_grad()
		actor_loss_v.backward()
		agent.actor_optimizer.step()

		self.actor_loss.append(actor_loss_v.item())

		# ------------------- update target network ------------------- #
		agent.soft_update(agent.critic_local, agent.critic_target, self.tau)
		agent.soft_update(agent.actor_local, agent.actor_target, self.tau)


	def distr_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma, device):
		"""
		from deep reinforcement learning Hands-on (Distributional Policy Gradients) pag 522
		https://arxiv.org/abs/1707.06887

		:param next_distr_v:
		:type next_distr_v:
		:param rewards_v:
		:type rewards_v:
		:param dones_mask_t:
		:type dones_mask_t:
		:param gamma:
		:type gamma:
		:param device:
		:type device:
		:return:
		:rtype:
		"""
		next_distr = next_distr_v.data.cpu().numpy()
		rewards = rewards_v.data.cpu().numpy()
		dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
		batch_size = len(rewards)
		proj_distr = np.zeros((self.batch_size, self.n_atoms), dtype=np.float32)

		for atom in range(self.n_atoms):
			tz_j = np.minimum(self.vmax, np.maximum(self.vmin, rewards + (self.vmin + atom * self.delta_z) * gamma))
			b_j = (tz_j - self.vmin) / self.delta_z
			l = np.floor(b_j).astype(np.int64)
			u = np.ceil(b_j).astype(np.int64)
			eq_mask = u == l
			proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
			ne_mask = u != l
			proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
			proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

		if dones_mask.any():
			proj_distr[dones_mask] = 0.0
			tz_j = np.minimum(self.vmax, np.maximum(self.vmin, rewards[dones_mask]))
			b_j = (tz_j - self.vmin) / self.delta_z
			l = np.floor(b_j).astype(np.int64)
			u = np.ceil(b_j).astype(np.int64)
			eq_mask = u == l
			eq_dones = dones_mask.copy()
			eq_dones[dones_mask] = eq_mask
			if eq_dones.any():
				proj_distr[eq_dones, l[eq_mask]] = 1.0
			ne_mask = u != l
			ne_dones = dones_mask.copy()
			ne_dones[dones_mask] = ne_mask
			if ne_dones.any():
				proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
				proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
		return torch.FloatTensor(proj_distr).to(device)

	def step(self, states, actions, rewards, next_states, dones):
		"""
		add experience to the buffer per each agent and call learn if memory > batch size

		:param states (n_agents, state_size) (numpy): agents' state of current timestamp
		:param actions (n_agents, action_size) (numpy): agents' action of current timestamp
		:param rewards (n_agents,):
		:param next_states (n_agents, state_size) (numpy):
		:param dones (n_agents,) (numpy):
		:return:
		"""

		self.memory.add(states, actions, rewards, next_states, dones)

		# activate learning every few steps
		self.t_step = self.t_step + 1
		if self.t_step % self.learn_every_step == 0:
			# Learn, if enough samples are available in memory
			if len(self.memory) > self.batch_size:
				for _ in range(self.learn_repeat):
					b_a_states, b_a_actions, b_rewards, b_a_next_states, b_dones = self.memory.sample()

					j=0
					for i, agent in enumerate(self.mad4pg_agent):
						if i % 2 == 0:
							j=0
						states = b_a_states[:,j,:].squeeze(1) # (batch_size, state_size)
						actions = b_a_actions[:,j,:].squeeze(1) # (batch_size, action_size)
						rewards = b_rewards[:,j] # (batch_size,)
						next_states = b_a_next_states[:,j,:].squeeze(1) # (batch_size, next_states)
						dones = b_dones[:,j] # (batch_size,)

						self.learn(agent, states, actions, rewards, next_states, dones, self.gamma)
						j= j+1


def train_mad4pg(agent, n_agents, af, magents, n_episodes, env, brain_name, check_pth='./checkpoints/checkpoint.pth'):
	epi_scores = []
	scores_window = deque(maxlen=100)

	for i_episode in range(1, n_episodes + 1):
		env_info = env.reset(train_mode=True)[brain_name]
		states = env_info.vector_observations
		scores = np.zeros(n_agents)
		while True:
			actions = agent.acts(states, mode='train')  # (n_agents, action_size)
			env_info = env.step(actions)[brain_name]
			next_states = env_info.vector_observations
			rewards = env_info.rewards  # (n_agents,)
			dones = env_info.local_done  # (n_agents,)

			agent.step(states, actions, rewards, next_states, dones)

			scores += env_info.rewards  # (n_agents,)
			states = next_states  # (n_agents, state_size)
			if np.any(dones):
				break
		scores_window.append(np.max(scores))
		epi_scores.append(np.max(scores))
		print('\rEpisode {:>4}\tAverage Score:{:>6.3f}\tMemory Size:{:>5}'.format(
			i_episode, np.mean(scores_window), len(agent.memory)), end="")
		if i_episode % 1000 == 0:
			print('\rEpisode {:>4}\tAverage Score:{:>6.3f}\tMemory Size:{:>5}'.format(
				i_episode, np.mean(scores_window), len(agent.memory)))
		if np.mean(scores_window) > 0.8:
			break
	checkpoint= {}
	for i in range(magents):

		checkpoint[f'actor{i}'] = agent.mad4pg_agent[i].actor_local.state_dict()
		checkpoint[f'critic{i}'] = agent.mad4pg_agent[i].critic_local.state_dict()


	check_pth = f'./checkpoints/checkpoint_{magents}_{af}.pth'
	torch.save(checkpoint, check_pth)
	return epi_scores, agent.actor_loss, agent.critic_loss

def play_agent(agent, n_agents, af, path='./checkpoints/checkpoint.pth'):
	"""
	Play Agent
	:param agent:
	:type agent:
	:param n_agents:
	:type n_agents:
	:param af:
	:type af:
	:param path:
	:type path:
	:return:
	:rtype:
	"""
	path = f'./checkpoints/checkpoint_{n_agents}_{af}.pth'
	checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
	for i in range(n_agents):
		actor0_state_dict = checkpoint[f'actor{i}']
		critic0_state_dict = checkpoint[f'critic{i}']

		agent.mad4pg_agent[i].actor_local.load_state_dict(actor0_state_dict)
		agent.mad4pg_agent[i].critic_local.load_state_dict(critic0_state_dict)







