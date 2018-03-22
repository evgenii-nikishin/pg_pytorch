import numpy as np
from src.utils import TrajStats
from multiprocessing import Process, Pipe

def SubprocEnvs_worker(remote, parent_remote, env_wrapper):
    parent_remote.close()
    env = env_wrapper
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except Exception as e:
        print (e.message)
        remote.close()

class SubprocEnvs:
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.ind_alive = np.empty(0)
        self._init_processes(env_fns, SubprocEnvs_worker)

    def _init_processes(self, env_fns, worker):
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes = np.array(self.remotes)

    def __len__(self):
        return self.nenvs

    def has_alive_envs(self):
        return len(self.ind_alive) > 0

    def get_indices_alive(self):
        return self.ind_alive

    def update_ind_alive(self, dones):
        self.ind_alive = self.ind_alive[np.logical_not(dones)]

    def step(self, actions):
        assert len(actions) == len(self.ind_alive), \
            'The number of actions doesn\'t equal to the number of alive agents.'
        for remote, action in zip(self.remotes[self.ind_alive], actions):
            remote.send(('step', action[0]))
        
        results = [remote.recv() for remote in self.remotes[self.ind_alive]]
        obs, rews, dones, infos = zip(*results)
        self.update_ind_alive(dones)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        self.ind_alive = np.arange(self.nenvs)
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
