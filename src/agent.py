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


class AgentA2C(nn.Module):
    def __init__(self, env):
        """
        Constructor
        Arguments:
            env          --  environment
            gamma        --  discount factor, float
            lr           --  learning rate, float
            save_returns --  whether save returns on each steps, bool
        """

        super(AgentA2C, self).__init__()
        self.env = env
        
        #self.is_cuda = False
        self.n_actions = int(env.action_space.n)

        # init of AC nets depends on type of environment
        if type(env.observation_space) == gym.spaces.Discrete:
            self.s_shape = (env.observation_space.n, )
            self.net = FCNet(self.s_shape[0], self.n_actions, need_encode=True)
        elif type(env.observation_space) == gym.spaces.Box and len(env.observation_space.shape) == 1:
            self.s_shape = (env.observation_space.shape[0], )
            self.net = FCNet(self.s_shape[0], self.n_actions, need_encode=False)
        elif type(env.observation_space) == gym.spaces.Box and len(env.observation_space.shape) == 3:
            self.s_shape = env.observation_space.shape
            self.net = ConvNets(self.s_shape, self.n_actions)
        else:
            raise ValueError('Unknown observation space')
        
        self.optimizer = optim.Adam(self.net.parameters())
    '''    
    def cuda(self):
        """
        Move model to GPU
        """
        self.is_cuda = True
        self.net.cuda()
        return super(AgentA2C, self).cuda()
    
    def cpu(self):
        """
        Move model to CPU
        """
        self.is_cuda = False
        self.net.cpu()
        return super(AgentA2C, self).cpu()
    '''
    def forward(self, state):
        """
        Computes logits of pi(a | s) and V(s)
        Arguments:
            state   -- state for which actions distr. need to be computed
        """
        
        state_enc = Variable(self.net.encode_state(state))
        return self.net(state_enc)
        
    def act(self, state):
        """
        Samples from pi(a | s)
        Arguments:
            state   -- state from which agent acts
        """

        logits, _ = self.forward(state)
        return torch.multinomial(F.softmax(logits, dim=-1), 1).data[0]
    
    def sample_action(self, logits):
        """
        Added in order not to do forward pass two times during learning
        Arguments:
            logits  -- output of pi_head
        """
        return torch.multinomial(F.softmax(logits, dim=-1), 1).data[0]
    
    def get_policy(self):
        """
        Returns pi(a | s) for all possible states
        """

        if type(self.env.observation_space) != gym.spaces.Discrete:
            raise ValueError('Avaliable only for discrete state spaces')

        all_states = np.arange(self.s_shape[0])
        states = Variable(self.net.encode_states(all_states))
        logits, _ = self.net(states)
        return F.softmax(logits, dim=-1).cpu().data.numpy()