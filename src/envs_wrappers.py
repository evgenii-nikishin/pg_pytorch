import numpy as np
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper#.x()
    try:
	    while True:
	        cmd, data = remote.recv()
	        if cmd == 'step':
	            ob, reward, done, info = env.step(data)
	            if done:
	                ob = env.reset()
	            remote.send((ob, reward, done, info))
	        elif cmd == 'reset':
	            ob = env.reset()
	            remote.send(ob)
	        elif cmd == 'close':
	            remote.close()
	            break
	        #elif cmd == 'get_spaces':
	        #    remote.send((env.observation_space, env.action_space))
	        else:
	            raise NotImplementedError
    except Exception as e:
        print (e.message)
        remote.close()

class SubprocAsyncEnvs:
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn)) #CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True