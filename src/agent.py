import gym
import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from src.networks import FCNet
from src.utils import TrajStats


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        """
        Constructor
        Arguments:
            gamma        --  discount factor, float
            lr           --  learning rate, float
            save_returns --  whether save returns on each steps, bool
        """

        super(Agent, self).__init__()
        
        self.observation_space = observation_space
        self.n_actions = action_space.n

        # init of AC nets depends on type of environment
        if type(observation_space) == gym.spaces.Discrete:
            self.s_shape = (observation_space.n, )
            self.net = FCNet(self.s_shape[0], self.n_actions, need_encode=True)
        elif type(observation_space) == gym.spaces.Box and len(observation_space.shape) == 1:
            self.s_shape = (observation_space.shape[0], )
            self.net = FCNet(self.s_shape[0], self.n_actions, need_encode=False)
        elif type(observation_space) == gym.spaces.Box and len(observation_space.shape) == 3:
            self.s_shape = observation_space.shape
            self.net = ConvNets(self.s_shape, self.n_actions)
        else:
            raise ValueError('Unknown observation space')
    
    def cuda(self):
        """
        Move model to GPU
        """

        self.net.is_cuda = True
        return super(Agent, self).cuda()

    def cpu(self):
        """
        Move model to CPU
        """
        
        self.net.is_cuda = False
        return super(Agent, self).cpu()

    def forward(self, states):
        """
        Computes logits of pi(a | s) and V(s)
        Arguments:
            states   -- states for which actions distr. need to be computed
        """
        
        state_enc = Variable(self.net.encode_states(states))
        
        return self.net(state_enc)
        
    def act(self, states):
        """
        Samples from pi(a | s)
        Arguments:
            state   -- state from which agent acts
        """

        logits, _ = self.forward(states)
        return self.sample_action(logits)
    
    def sample_action(self, logits):
        """
        Added in order not to do forward pass two times during learning
        Arguments:
            logits  -- output of pi_head
        """

        return torch.multinomial(F.softmax(logits, dim=-1), 1).data
    
    def get_policy(self):
        """
        Returns pi(a | s) for all possible states
        """

        if type(self.observation_space) != gym.spaces.Discrete:
            raise ValueError('Avaliable only for discrete state spaces')

        all_states = np.arange(self.s_shape[0])
        states = Variable(self.net.encode_states(all_states))
        logits, _ = self.net(states)
        return F.softmax(logits, dim=-1).cpu().data.numpy()