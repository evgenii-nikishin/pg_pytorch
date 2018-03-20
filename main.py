import gym
import argparse
import numpy as np

import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from src.agent import Agent
from src.utils import TrajStats, set_seeds
from src.envs_wrappers import SubprocEnvs
from src.trpo import *


from src.kfac import KFAC_Optim

def learn(agent, envs, update_rule, n_timesteps=1e5, gamma=0.99, lambda_gae=0.97, entr_coef=1e-3, max_kl=1e-2, log_interval=1e4):
    """
    Optimize networks parameters via interacting with env
    Arguments:
        agent           --  agent to optimize
        envs            --  list of environments to interact with
        update_rule     --  'A2C', 'TRPO' or 'K-FAC', str
        n_timesteps     --  number of interactions with environments, int
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
    
    if update_rule == 'A2C':
        optimizer = optim.Adam(agent.parameters())
    elif update_rule == 'TRPO':
        optimizer = optim.Adam(agent.net.value_head.parameters())
    elif update_rule == 'K-FAC':
        optimizer = KFAC_Optim(agent, delta=max_kl)
    else:
        raise ValueError('Unknown update rule')

    while timestep < n_timesteps:
        states = w_envs.reset()
        tss = [TrajStats() for _ in range(n_envs)]

        while w_envs.has_alive_envs():
            logits, value = agent.forward(states)
            actions = agent.sample_action(logits)

            ind_alive = w_envs.get_indices_alive()
            states_new, rewards, done, _ = w_envs.step(actions)      
            
            for i, i_alive in enumerate(ind_alive):
                tss[i_alive].append(rewards[i], F.log_softmax(logits[i], dim=-1)[actions[i]], value[i], logits[i], states[i], actions[i])
            states = states_new[np.logical_not(done)]

            timestep_diff += len(ind_alive)
            timestep += len(ind_alive)
            if timestep_diff >= log_interval:
                timestep_diff -= log_interval
                print('{} timesteps, av. return: {:.3f}'.format((timestep // log_interval) * log_interval, 
                                                                np.mean(returns[-300:])))
    
        critic_loss = 0
        actor_loss = 0
        for ts in tss:
            episode_returns = ts.calc_episode_returns(gamma)
            critic_loss += 0.5*(ts.get_values() - episode_returns).pow(2).sum()
            returns.append(ts.calc_return(gamma))
            
            advantages = ts.calc_gaes(gamma, lambda_gae)
            logs_pi = ts.get_logs_pi_a()
            actor_loss += -(logs_pi * advantages.detach()).sum()  # minus added in order to ascend

        optimizer.zero_grad()
        
        if update_rule == 'A2C' or update_rule == 'K-FAC':
            ((actor_loss + critic_loss) / n_envs).backward()
        elif update_rule == 'TRPO':
            
            critic_flat_grads = get_flat_grads(agent, critic_loss/n_envs)
            flat_grads = get_flat_grads(agent, actor_loss).data
            
            if np.allclose(flat_grads.numpy(), 0):
                print('zero gradients, passing')
                continue

            kl = 0
            for ts in tss:
                logits = ts.get_logits()
                kl += compute_kl(logits, logits.detach())

            flat_grads_kl = get_flat_grads(agent, kl, support_next_order=True)
            hess_vec = lambda vec: hess_vec_full(vec, agent, flat_grads_kl, 1e-3)

            stepdir = cg(hess_vec, -flat_grads, cg_iters=10)
            shs = 0.5 * torch.dot(stepdir, hess_vec(stepdir))
            
            lm = np.sqrt(shs / max_kl)
            proposed_step = stepdir / lm
            neggdotstepdir = torch.dot(-flat_grads, stepdir)

            compute_obj = lambda flat_params: compute_obj_full(flat_params, agent, tss, gamma, lambda_gae)
            params_prev = get_flat_params(agent)
            success, params_new = linesearch(compute_obj, params_prev, proposed_step, neggdotstepdir / lm)
            set_flat_params(agent, params_new)

            set_flat_grads(agent, critic_flat_grads)

        optimizer.step()
    w_envs.close()


def main():
    print ("note: 'ulimit -Sn 1024' if Errno 24")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=417)
    parser.add_argument('--n-timesteps', type=int, default=1e5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max-kl', type=float, default=1e-2)
    parser.add_argument('--log-interval', type=int, default=1e4)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--update-rule', default='A2C')
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'No available cuda devices'

    envs = [gym.make(args.env) for _ in range(args.batch_size)]
    set_seeds(envs, args.seed, args.cuda)

    agent = Agent(envs[0].observation_space, envs[0].action_space)
    if args.cuda:
        agent.cuda()

    learn(agent, envs, args.update_rule, n_timesteps=args.n_timesteps, gamma=args.gamma, 
          log_interval=args.log_interval, max_kl=args.max_kl)
    if not (args.save_path is None):
        torch.save(agent.state_dict(), args.save_path)

if __name__ == '__main__':
    main()
