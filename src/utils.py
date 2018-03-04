import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Variable


def arr2var(x, cuda=False):
    var_x = Variable(torch.FloatTensor(x))
    return var_x.cuda() if cuda else var_x

def set_seeds(env, seed, is_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    if is_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class TrajStats:
    """
        Class to efficiently store and operate with trajectory's data
    """
    
   
    def __init__(self):
        self.rewards = []
        self.logs_pi_a = []
        self.values = []
        
    def _is_cuda(self):
        """
        Check if data located on CUDA
        Returns 'False' if no data available
        """
        
        return self.values[0].is_cuda if len(self.values) > 0 else False

    def append(self, r, log_pi_a, v):
        """
        Adds r(s_t, a_t), log pi(a_t | s_t), V(s_t)
        """

        self.rewards.append(r)
        self.logs_pi_a.append(log_pi_a)
        self.values.append(v)

    def get_values(self):
        """
        Returns value functions for each timestep
        """

        return torch.cat(self.values)
       
    def calc_return(self, gamma):
        """
        Calculates cumulative discounted rewards
        """

        y = 0
        for x in self.rewards[::-1]:
            y = gamma * y + x
        return y
    
    def calc_gaes(self, gamma, lambda_gae):
        """
        Calculates generalized advantage function for each timestep
        https://arxiv.org/abs/1506.02438
        """
        assert len(self.values) > 0, "Storage must be non-empty"
        
        zero = torch.zeros(1)
        rewards = torch.FloatTensor(self.rewards)
        if self._is_cuda():
            zero    = zero.cuda()
            rewards = rewards.cuda()
            
        next_v = torch.cat([torch.cat(self.values[1:]), Variable(zero)]).data \
                    if len(self.values) > 1 else zero
            
        target = Variable(rewards + gamma * next_v, requires_grad=False)
        deltas = target - torch.cat(self.values)
        lg = lambda_gae * gamma
        
        A = 0
        advantages = []
        for delta in reversed(deltas):
            A = A * lg + delta
            advantages.append(A)
        
        return torch.stack(advantages[::-1])

    def calc_episode_returns(self, gamma):
        """
        Calculates cumulative discounted rewards starting from each timestep
        """

        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = G * gamma + r
            returns.append(G)
            
        return arr2var(returns[::-1], cuda=self._is_cuda())

    def calc_advs(self, gamma, n_step=5):
        """
        Calculates n-step advantage function for each timestep
        """
        n_step = min(n_step, len(self.rewards))
        if len(self.values) > n_step:
            next_v = torch.cat([torch.cat(self.values[n_step:]), Variable(torch.zeros(n_step))]).data
        else:
            next_v = torch.zeros(n_step)
            
        rewards_list = []
        cur_gamma = 1
        for i in range(n_step):
            # comp_zeros = np.zeros(min(i + max(n_step - len(self.rewards), 0), n_step))
            rewards_list.append(np.concatenate((self.rewards[i:], np.zeros(i))) * cur_gamma)
            cur_gamma *= gamma
        
        target = Variable(torch.FloatTensor(np.sum(rewards_list, axis=0)) + cur_gamma * next_v, requires_grad=False)
        # advantages = target - torch.cat([torch.cat(self.values), Variable(torch.zeros(max(0, n_step - len(self.values))))])
        advantages = target - torch.cat(self.values)
        
        return advantages


# NOT FINISHED YET
class Experience:
    def __init__(self):
        self.trajectories = []
    
    def append(self, trajectory):
        self.trajectories.append(trajectory)
    
    def sample(self, batch_size=1):
        return np.random.choice(self.trajectories, size=batch_size)

    def get_last(self, batch_size=1):
        return self.trajectories[-batch_size:]


# NOT FINISHED YET
class ReshapingLayer(nn.Module):
    """
        Wrapper for 'reshape' 
        to embed this operation in nn.Sequential(...)
    """

    def __init__(self, *args):
        """
        Constructor
        Arguments:
            *args  -- new shape dimensions
        Example:
            ReshapingLayer(10, 20, -1)
        """

        super(ReshapingLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        """
        Reshape input w.r.t. class parameters
        Arguments:
            x   --  input data, tensor
        """

        return x.view(self.shape)
