### Library with implementations of most popular Policy Gradient methods for Reinforcement Learning

The code is organized as follows:
* **src/agent.py**: implementations of Agent.
* **src/env_wrappers.py**: wrapper for multiple parallel environments
* **src/kfac.py**: implementation of K-FAC optimizer, compatible with `torch.optim.Optimizer`
* **src/networks.py**: neural network architectures of actor and critic for different environments
* **src/trpo.py**: implementation of TRPO-optimizer routines 
* **src/utils.py**: utils for models and optimizers


#### What do we have:
* A2C
* TRPO
* ACKTR

#### What can we add later:
* PPO
* A3C

#### TODOs

* effective vectorization with n-step returns
* more detailed documentation

------------------------------------------

Code is developed and supported by:
* Eugenii Nikishin [nikishin-evg](https://github.com/nikishin-evg) (*nikishin.evg@gmail.com*)
* Iurii Kemaev [hbq1](https://github.com/hbq1) (*y.kemaev@gmail.com*)
* Maxim Kuznetsov [binom16](https://github.com/binom16) (*binom16@gmail.com*)

------------------------------------------

Inherited from [https://github.com/nikishin-evg/acktr_pytorch](https://github.com/nikishin-evg/acktr_pytorch)
