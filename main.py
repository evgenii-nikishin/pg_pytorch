import gym
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.agent import AgentA2C
from src.utils import TrajStats, set_seeds
from src.envs_wrappers import SubprocEnvs


def learn(agent, envs, optimizer, n_timesteps=1e5, gamma=0.99, lambda_gae=0.95, entr_coef=1e-3, log_interval=1e4):
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
    n_envs = len(envs)
    w_envs = SubprocEnvs(envs)

    agent.train()
    returns = []
    timestep = 0
    timestep_diff = 0
    episode = 0

    while timestep < n_timesteps:
        states_alive = w_envs.reset()
        tss = [TrajStats() for _ in range(n_envs)]

        episode += 1
        while w_envs.has_alive_envs():
            logits, value = agent.forward(states_alive)
            actions = agent.sample_action(logits)

            ind_alive = w_envs.get_indices_alive()
            states_new, rewards, done, _ = w_envs.step(actions)
            
            for i, i_alive in enumerate(ind_alive):
                tss[i_alive].append(rewards[i], F.log_softmax(logits[i], dim=-1)[actions[i]], value[i], logits[i])
            states_alive = states_new[np.logical_not(done)]

            timestep_diff += len(ind_alive)
            timestep += len(ind_alive)
            if timestep_diff >= log_interval:
                timestep_diff -= log_interval
                print('{} timesteps, av. return: {:.3f}'.format((timestep // log_interval) * log_interval, np.mean(returns[-50:])))

        loss = Variable(torch.Tensor([0]))
        for ts in tss:
            traj_loss, traj_return = ts.calc_loss(gamma, lambda_gae)
            returns.append(traj_return)
            loss += traj_loss

        optimizer.zero_grad()
        (loss / n_envs).backward()
        optimizer.step()

    w_envs.close()


def main():
    print ("note: 'ulimit -Sn 1024' if Errno 24")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='FrozenLake-v0')
    parser.add_argument('--seed', type=int, default=417)
    parser.add_argument('--n-timesteps', type=int, default=1e5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--log-interval', type=int, default=1e4)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cuda', type=bool, default=False)
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'No available cuda devices'

    envs = [ gym.make(args.env) for _ in range(args.batch_size) ]
    set_seeds(envs, args.seed, args.cuda)

    agent = AgentA2C(envs[0].observation_space, envs[0].action_space)
    if args.cuda:
        agent.cuda()

    optimizer = torch.optim.Adam(agent.parameters())
    learn(agent, envs, optimizer, n_timesteps=args.n_timesteps, gamma=args.gamma, log_interval=args.log_interval, lambda_gae=0.99)
    if not (args.save_path is None):
        torch.save(agent.state_dict(), args.save_path)

if __name__ == '__main__':
    main()
