### A library with implementations of popular Policy Gradient methods for Reinforcement Learning

#### Includes the following algorithms:
* [A2C](https://arxiv.org/abs/1602.01783)
* [TRPO](https://arxiv.org/abs/1502.05477)
* [ACKTR](https://arxiv.org/abs/1708.05144)


The code is organized as follows:
* **src/agent.py**: implementations of Agent.
* **src/env_wrappers.py**: wrapper for multiple parallel environments
* **src/kfac.py**: implementation of K-FAC optimizer, compatible with `torch.optim.Optimizer`
* **src/networks.py**: neural network architectures of actor and critic for different environments
* **src/trpo.py**: implementation of TRPO-optimizer routines 
* **src/utils.py**: utils for models and optimizers


#### TODO:

* effective vectorization with n-step returns
* PPO
* A3C

------------------------------------------

Code is developed and supported by:
* Evgenii Nikishin [nikishin-evg](https://github.com/nikishin-evg) (*nikishin.evg@gmail.com*)
* Iurii Kemaev [hbq1](https://github.com/hbq1) (*y.kemaev@gmail.com*)
* Maxim Kuznetsov [binom16](https://github.com/binom16) (*binom16@gmail.com*)

------------------------------------------

Inherited from [https://github.com/nikishin-evg/acktr_pytorch](https://github.com/nikishin-evg/acktr_pytorch)
