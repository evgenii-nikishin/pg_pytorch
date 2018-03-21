import numpy as np
from skimage.transform import resize

import torch
from torch import nn
from torch.autograd import Variable

from src.utils import Flatten

class FCNet(nn.Module):
    """
        Wrapper for Actor and Critic neural networks.
        Use this in games with one-dimensional or discrete state space.
    """

    def __init__(self, n_states, n_actions, need_encode=True):
        """
        Constructor
        Arguments:
            n_states    - number of possible states, int 
            n_actions   - number of possible actions, int 
            need_encode - whether to make one-hot encoding for states, bool 
        """

        super(FCNet, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.need_encode = need_encode
        
        self.is_cuda = False
        self._init_architecture()
        
    def _init_architecture(self):
        """
        Initialize actor and critic neural network
        """

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.n_states, 64))
        self.act = nn.Tanh()
        
        self.pi_head = nn.Linear(64, self.n_actions)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Compute pi(a | s) and V(s)
        """
        
        for layer in self.layers:
            x = self.act(layer(x))
        return self.pi_head(x), self.value_head(x)
    
    def encode_states(self, states):
        """
        One-hot encode group of states
        Arguments:
            states  -- group of states to encode
        """

        if self.need_encode:
            n = len(states)
            encoded = torch.zeros(n, self.n_states)
            indices = np.array(states)
            #    states[:,0] if len(states.shape) > 1 else states
            #)
            encoded[np.arange(n), indices] = 1.0
        else:
            encoded = torch.FloatTensor(states)
            
        if self.is_cuda:
            encoded = encoded.cuda()
        return encoded


# NOT FINISHED YET
class ConvNet(nn.Module):
    """
        Wrapper for Actor and Critic neural networks.
        Use this in games with image state space, e.g. Atari 2600
    """

    def __init__(self, n_states, n_actions, inner_state_shape=(84,84)):
        """
        Constructor
        Arguments:
            n_states            -- number of possible states, int 
            n_actions           -- number of possible actions, int 
            inner_state_shape   -- shape of state represesentation in NNs, tuple
        """

        super(ConvNet, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.inner_state_shape = inner_state_shape

        self.is_cuda = False
        # manual parameter, depends on architecture
        self.hidden_dim = 32 * 7 * 7

        self._init_net_()

    def _init_net_(self):
        """
        Initialize actor neural network
        """

        self.common_part = nn.Sequential(
            nn.Conv2d(self.n_states[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.pi_head = nn.Sequential(nn.Linear(32 * 7 * 7, 64), nn.Tanh(), nn.Linear(64, self.n_actions))
        self.value_head = nn.Sequential(nn.Linear(32 * 7 * 7, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        """
        Compute pi(a | s) and V(s)
        """

        x = self.encode_states(x)

        if type(x) is not Variable:
            x = Variable(x)

        common = self.common_part(x)
        common = common.view(common.shape[0], -1)


        return self.pi_head(common), self.value_head(common)

    def cuda(self):
        """
        Move nets to GPU
        """

        self.is_cuda = True
        return super(ConvNet, self).cuda()
    
    def cpu(self):
        """
        Move nets to CPU
        """

        self.is_cuda = False
        return super(ConvNet, self).cpu()
    
    def preprocess_img(self, imgs):
        """
        Resize image for universal representation of env. state
        Arguments:
            imgs    -- array of images to resize 
        """

        if type(imgs) is not np.ndarray:
            imgs = imgs.data.numpy().copy()

        if imgs.shape[1:] != self.inner_state_shape:
            return torch.from_numpy(resize(imgs.astype(np.float), self.inner_state_shape, mode='reflect')).float()
        else:
            return torch.from_numpy(imgs.astype(np.float)).float()
    
    def encode_state(self, s):
        """
        One-hot encode state
        Arguments:
            state   -- state to encode
        """

        enc = torch.FloatTensor(self.preprocess_img(s)).view((-1, self.n_states[2]) + self.inner_state_shape)

        if self.is_cuda:
            enc = enc.cuda()
        return enc
    
    def encode_states(self, states):
        """
        One-hot encode group of states
        Arguments:
            states  -- group of states to encode
        """
        # NOTE: can be optimized

        ans = torch.cat([self.encode_state(s) for s in states])
        if type(ans) is Variable:
            ans = ans.data

        return ans
