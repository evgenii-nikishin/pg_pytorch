import gym
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from agent import AgentA2C
from utils import TrajStats, set_seeds


def learn(agent, env, n_episodes=20000, gamma=0.99, lambda_gae=0.95, entr_coef=1e-3, verbose=0):
    """
    Optimize networks parameters via interacting with env
    Arguments:
        agent           --  agent to optimize
        env             --  environment to interact with
        n_episodes      --  number of full interaction emulations, int
        lambda_gae      --  mixing coefficient in generalized advantage estimation
        entr_coef       --  entropy loss multiplier, float
        verbose         --  number of episodes to print debug info, int (default is 0: don't print)
    """

    agent.net.train()

    returns = []
    for e in range(n_episodes):
        s = env.reset()
        done = False
        ts = TrajStats()
        
        while not done:
            logits, value = agent.forward(s)
            a = agent.sample_action(logits)
            s_new, r, done, _ = env.step(a)
            
            ts.append(r, F.log_softmax(logits, dim=-1)[a], value)
            s = s_new

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

        if verbose > 0 and (e+1) % verbose == 0:
            print('{} episodes, av. return: {:.3f}'.format(e+1, np.mean(returns[-50:])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='FrozenLake-v0')
    parser.add_argument('--seed', type=int, default=417)
    parser.add_argument('--n-episodes', type=int, default=20000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--verbose', type=int, default=1000)
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = AgentA2C(env)

    set_seeds(env, args.seed)
    learn(agent, env, n_episodes=args.n_episodes, gamma=args.gamma, verbose=args.verbose)


if __name__ == '__main__':
    main()