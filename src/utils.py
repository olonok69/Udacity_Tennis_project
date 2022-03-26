import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, namedtuple
import random
import numpy as np
import torch
from unityagents import UnityEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_env(worker_id, base_port, file="Tennis_Windows_x86_64/Tennis.exe",
             grap=True, train=True):
    """
    load Unity Environment
    :param worker_id: ID env
    :param base_port: communications port with unity agent
    :param file: Unity executable
    :param grap: If to show graphs or not. Typically in training I hide graphics
    :param train: if train mode or test mode
    """
    # load environtment
    env = UnityEnvironment(file_name=file, worker_id=worker_id, base_port=base_port, no_graphics=grap)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # reset the environment
    env_info = env.reset(train_mode=train)[brain_name]


    state = env_info.vector_observations[0]
    state_size = len(state)
    return env , brain_name, brain, action_size, env_info, state, state_size

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a new experience to memory.

        :param
            states (n_agents, state_size) (numpy)
            actions (n_agents, action_size) (numpy)
            rewards (n_agents,) (numpy)
            next_states (n_agents, state_size) (numpy)
            dones (n_agents,) (numpy)
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.

        :return
            b_a_states (batch_size, n_agents, state_size) (tensor)
            b_a_actions (batch_size, n_agents, action_size) (tensor)
            b_rewards (batch_size, n_agents) (numpy)
            b_a_next_states (batch_size, n_agents, next_states) (tensor)
            b_dones (batch_size, n_agents) (tensor)

        """
        experiences = random.sample(self.memory, k=self.batch_size)

        b_a_states = torch.from_numpy(np.vstack([np.expand_dims(e.states, axis=0) for
                                                 e in experiences if e is not None])).float().to(device)
        b_a_actions = torch.from_numpy(np.vstack([np.expand_dims(e.actions, axis=0) for
                                                  e in experiences if e is not None])).float().to(device)
        b_rewards = torch.from_numpy(np.vstack([np.expand_dims(e.rewards, axis=0) for
                                                e in experiences if e is not None])).float().to(device)
        b_a_next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_states, axis=0)
                                                      for e in experiences if e is not None]))\
            .float().to(device)
        b_dones = torch.from_numpy(np.vstack([np.expand_dims(e.dones, axis=0) for e in experiences if
                                              e is not None]).astype(np.uint8)).float().to(
            device)

        return b_a_states, b_a_actions, b_rewards, b_a_next_states, b_dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def plot_scores(scores,
                nagents,
                activation_function,
                rolling_window=100):
    '''
    Plot score and its moving average on the same chart.'''

    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores, '-y', label='episode score')
    plt.title(f'Score and rolling mean. Number Agents {str(nagents)}, Activation Function {str(activation_function)}')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(np.arange(len(scores)), rolling_mean, '-r', label='rolling_mean')
    plt.ylabel('score')
    plt.xlabel('episode #')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'images/scores_all_{nagents}_{activation_function}.jpg')
    return

def plot_critic_loss(scores,
                nagents,
                activation_function,
                rolling_window=100):
    '''
    Plot critic loss and its moving average on the same chart.'''

    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores, '-y', label='episode score')
    plt.title(f'Criticc Loss and rolling mean. Number Agents {str(nagents)}, '
              f'Activation Function {str(activation_function)}')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(np.arange(len(scores)), rolling_mean, '-r', label='rolling_mean')
    plt.ylabel('loss')
    plt.xlabel('episode #')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'images/critic_loss_{nagents}_{activation_function}.jpg')
    return

def plot_actor_loss(scores,
                nagents,
                activation_function,
                rolling_window=100):
    '''
    Plot actor loss and its moving average on the same chart.'''

    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores, '-y', label='episode score')
    plt.title(f'Actor Loss and rolling mean. Number Agents {str(nagents)}, '
              f'Activation Function {str(activation_function)}')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(np.arange(len(scores)), rolling_mean, '-r', label='rolling_mean')
    plt.ylabel('loss')
    plt.xlabel('episode #')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'images/actor_loss_{nagents}_{activation_function}.jpg')
    return

