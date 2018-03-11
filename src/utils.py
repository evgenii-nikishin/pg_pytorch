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
    if is_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    mod = max(seed, 1e3) ** 2
    for e in env:
        e.seed(np.random.randint(mod))

    # reset seed for reproducibility without dependency on num. of envs
    np.random.seed(seed)


class TrajStats:
    """
        Class to efficiently store and operate with trajectory's data
    """
    
   
    def __init__(self):
        self.clear()
     
    def _is_cuda(self):
        """
        Check if data located on CUDA
        Returns 'False' if no data available
        """
        
        return self.values[0].is_cuda if len(self.values) > 0 else False

    def clear(self):
        self.rewards = []
        self.logs_pi_a = []
        self.values = []
        self.logits = []
        self.states = []
        self.actions = []

    def append(self, r, log_pi_a, v, logits, s, a):
        """
        Adds r(s_t, a_t), log pi(a_t | s_t), V(s_t), s_t, a_t
        """

        self.rewards.append(r)
        self.logs_pi_a.append(log_pi_a)
        self.values.append(v)
        self.logits.append(logits)
        self.states.append(s)
        self.actions.append(a)

    def get_values(self):
        """
        Returns value functions for each timestep
        """

        return torch.cat(self.values)

    def get_logs_pi_a(self):
        """
        Returns logs of prob of taken action for each timestep
        """

        return torch.cat(self.logs_pi_a)

    def get_logits(self):
        """
        Returns logits of pi(a | s) for each timestep
        """
        return torch.cat(list(map(lambda x: x.view(1, -1), self.logits)))

    def get_sar(self):
        """
        Returns sequence of states, actions and rewards (i.e. trajectory)
        TODO: verify for continuous and 3d states
        """
        return zip(self.states, self.actions, self.rewards)
       
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
                    if len(self.values) > 1 else Variable(zero)
            
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

    def calc_loss(self, gamma, lambda_gae):
        advantages = self.calc_gaes(gamma, lambda_gae)
        episode_returns = self.calc_episode_returns(gamma)
        logs_pi = torch.cat(self.logs_pi_a)
        return_val = self.calc_return(gamma)

        #entropy      = -(aprobs_var * torch.exp(aprobs_var)).sum()
        non_diff_advs = Variable(advantages.data, requires_grad=False)
        actor_loss   = -(logs_pi * non_diff_advs).sum()  # minus added in order to ascend

        #critic_loss  = 0.5*advantages.pow(2).sum()
        critic_loss  = 0.5*(self.get_values() - episode_returns).pow(2).sum()

        loss = actor_loss + critic_loss
        return loss, return_val

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
