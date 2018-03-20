import numpy as np
from src.utils import TrajStats

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def compute_kl(logits_1, logits_2):
    """
    Computes KL divergence between discrete distributions
    """
    probs_1 = F.softmax(logits_1, dim=-1)
    kl_components = probs_1 * (F.log_softmax(logits_1, dim=-1) - F.log_softmax(logits_2, dim=-1))
    return torch.mean(torch.sum(kl_components, dim=1))


def get_flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def set_flat_params(model, flat_params):
    ind_start = 0
    for param in model.parameters():
        ind_end = ind_start + np.prod(param.shape)
        param.data.copy_(flat_params[ind_start : ind_end].view(param.shape))
        ind_start = ind_end
        
def set_flat_grads(model, flat_grads):
    ind_start = 0
    for param in model.parameters():
        ind_end = ind_start + np.prod(param.shape)
        param.grad = flat_grads[ind_start : ind_end].view(param.shape)
        ind_start = ind_end
        
def get_flat_grads(model, loss, support_next_order=False):
    """
    Walkaround for computing grads in case loss does not depend on some leafs
    TODO: remove `try` later
    """
    
    if support_next_order:
        grads = []
        for param in model.parameters():
            try:
                grads.append(torch.autograd.grad(loss, param, create_graph=True)[0])
            except RuntimeError:
                grads.append(Variable(torch.zeros_like(param.data)))
    else:
        for p in model.parameters():
            p.grad = None

        loss.backward(retain_graph=True)
        grads = [p.grad if p.grad is not None else Variable(torch.zeros_like(p.data))
                 for p in model.parameters()]
        
    return torch.cat([grad.view(-1) for grad in grads])
        
def cg(matvec, b, cg_iters=10, residual_tol=1e-10):
    """
    Solves system Ax=b via conjugate gradients method.
    Adapted from John Schulman's code:
    https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
    Arguments:
        matvec        --  matrix-vector product function
        b             --  right-hand side
        cg_iters      --  number of iterations
        residual_tol  --  tolerance
    """
    
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for i in range(cg_iters):
        Ap = matvec(p)
        alpha = rdotr / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        newrdotr = torch.dot(r, r)
        beta = newrdotr / rdotr
        p = r + beta * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
            
    return x

def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.05):
    """
    Backtracking linesearch for finding optimal proposed step size.
    Adapted from John Schulman's code:
    https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
    Arguments:
        f                      --  
        x                      --  
        fullstep               --  
        expected_improve_rate  --  
        max_backtracks         --
        accept_ratio           --
    """
    fval = f(x)
    x_best = None
    max_ratio = -1
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = (newfval - fval).data[0]
        if actual_improve > 0:
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and ratio > max_ratio:
                max_ratio = ratio
                x_best = xnew
        
    return (True, x_best) if x_best is not None else (False, x) 


def hess_vec_full(vec, model, grads, damping):
    grads_vec = torch.dot(grads, Variable(vec))
    res = get_flat_grads(model, grads_vec).data

    return res + damping * vec


def compute_obj_full(flat_params, agent, tss, gamma, lambda_gae):
    # TODO: rewrite without new TrajStats 
    # TODO: and probably rewrite TrajStats, e.g. append to dict, make calc_gae not a method of class
    
    set_flat_params(agent, flat_params)
    
    res = 0
    
    for ts in tss:
        cur_ts = TrajStats()
        cur_ts.rewards = ts.rewards
        cur_ts.states = ts.states
        cur_ts.actions = ts.actions
        
        cur_ts.logits, cur_ts.values = agent.forward(cur_ts.states)
        cur_ts.values = [v for v in cur_ts.values]
        cur_ts.logs_pi_a = F.log_softmax(cur_ts.logits, dim=-1)[np.arange(len(cur_ts.actions)), 
                                                                np.array(cur_ts.actions)]
        cur_ts.logs_pi_a = [l for l in cur_ts.logs_pi_a]
        
        advantages = cur_ts.calc_gaes(gamma, lambda_gae)
        #advantages = cur_ts.calc_advs(gamma, n_step=1)
        old_logs_pi = ts.get_logs_pi_a()
        logs_pi = cur_ts.get_logs_pi_a()
        res += (torch.exp(logs_pi - old_logs_pi) * advantages.detach()).sum()
    
    return res / len(tss)