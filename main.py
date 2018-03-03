import gym
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.agent import AgentA2C
from src.utils import TrajStats, set_seeds


def learn(agent, env, n_timesteps=1e5, gamma=0.99, lambda_gae=0.95, entr_coef=1e-3, log_interval=1e4):
    """
    Optimize networks parameters via interacting with env
    Arguments:
        agent           --  agent to optimize
        env             --  environment to interact with
        n_episodes      --  number of full interaction emulations, int
        lambda_gae      --  mixing coefficient in generalized advantage estimation
        entr_coef       --  entropy loss multiplier, float
        log_interval    --  number of timesteps to print debug info, int
    """

    agent.net.train()
    returns = []
    timestep = 0
    episode = 0

    while timestep < n_timesteps:
        s = env.reset()
        done = False
        ts = TrajStats()
        
        episode += 1
        while not done:
            logits, value = agent.forward(s)
            a = agent.sample_action(logits)
            s_new, r, done, _ = env.step(a)
            
            ts.append(r, F.log_softmax(logits, dim=-1)[a], value)
            s = s_new
            timestep += 1
            if timestep % log_interval == 0:
                print('{} timesteps, av. return: {:.3f}'.format(timestep, np.mean(returns[-50:])))
        
        advantages = ts.calc_gaes(gamma, lambda_gae)
        episode_returns = ts.calc_episode_returns(gamma)
        logs_pi = torch.cat(ts.logs_pi_a)
        returns.append(ts.calc_return(gamma))

        #entropy      = -(aprobs_var * torch.exp(aprobs_var)).sum()
        non_diff_advs = Variable(advantages.data, requires_grad=False)
        actor_loss   = -(logs_pi * non_diff_advs).sum()  # minus added in order to ascend

        #critic_loss  = 0.5*advantages.pow(2).sum()
        critic_loss  = 0.5*(ts.get_values() - episode_returns).pow(2).sum()

        agent.optimizer.zero_grad()
        loss = actor_loss + critic_loss# - entr_coef * entropy
        loss.backward()
        agent.optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='FrozenLake-v0')
    parser.add_argument('--seed', type=int, default=417)
    parser.add_argument('--n-timesteps', type=int, default=1e5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--log_interval', type=int, default=1e4)
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = AgentA2C(env)

    set_seeds(env, args.seed)
    learn(agent, env, n_timesteps=args.n_timesteps, gamma=args.gamma, log_interval=args.log_interval)


if __name__ == '__main__':
    main()