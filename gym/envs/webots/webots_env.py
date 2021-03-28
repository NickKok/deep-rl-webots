"""
Gym Environment for the Webots simulator (www.cyberbotics.com)

Author: Florin Dzeladini
"""
import os

import subprocess
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import copy

import sys
import os
sys.path.append(os.path.abspath("__GLOBALPATH__/biorob-rl-communication-lib"))

try:
    from simplerpc import WebotsCommunicatorService
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install Webots_py, and also perform the setup instructions here: https://gitlab.com/srill-fb99/deep-rl-biorob.)".format(e))

WEBOTS_PATH = "/usr/local/bin/webotsNoStop"

class WebotsEnv(gym.Env):
    """Superclass for all Webots environments.
    """
    process_pid = 0
    def __init__(self, cfg, frame_skip,port,world=None):
        #TODO SEND RANDOM SEED TO WEBOTS.
        self.world = world
        self._launch_webots()
        self.frame_counter = 1
        self.frame_skip= frame_skip
        self.model = WebotsCommunicatorService(cfg['obs_dim'],cfg['action_space'],port)
        self.metadata = {
            'render.modes': ['human','bot'],
            'video.frames_per_second' : int(np.round(1.0 / self.model.opt['timestep']))
        }
        self.obs_dim = cfg['obs_dim']

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()



    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_pid(self):
        command1="ps -p {}".format(self.process_pid); 
        p2=subprocess.Popen(command1.split(),stdout=subprocess.PIPE); 
        return len(p2.communicate()[0].split('\n'))==3

    def _launch_webots(self):
        # We launch self.world
        if self.world:
            FNULL = open(os.devnull, 'w')
            p = subprocess.Popen([WEBOTS_PATH,self.world])
            self.process_pid = p.pid

    # methods to override:
    # ----------------------------
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        self.model.reset_model = True
        self.model.update()
        self.model.reset_model = False
        return self.state_vector()

    def reset(self):
        """
        Reset the robot degrees of freedom and return the state
        Implement this in each subclass.
        """
        self.model.reset_model = True
        self.model.update()
        self.model.reset_model = False
        self.model.update()
        self.model.update()
        return self.state_vector()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def set_state(self, qpos, qvel):
        # TODO Can be used to force the state.
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data['qpos'] = qpos
        self.model.data['qvel'] = qvel
        self.model.forceState()


    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def frame_length(self):
        if (self.frame_skip > 0):
            return self.frame_skip
        else:
            return self.frame_counter

    def do_one_simulation_step(self,ctrl):
        self.model.data['ctrl'] = ctrl
        self.model.update()

    def do_simulation(self, ctrl, n_frames):
        self.model.data['ctrl'] = ctrl
        for _ in range(n_frames):
            if not self.model.update():
                return False
        return True

    def _reset(self):
        ob = self.reset_model() # this returns the observations
        return ob

    def reset(self):
        ob = self._reset() # this returns the observations
        return ob

    def _render(self, mode='human', close=False):
        if mode == "human":
            self.model.fast_mode = False
        else:
            self.model.fast_mode = True
        pass

    def get_body_com(self, body_name):
        #TODO : Returns the Center Of Mass of a certain body
        # Question is it absolute coordinate or not ?
        # idx = self.model.body_names.index(six.b(body_name))
        # return self.model.data.com_subtree[idx]
        pass

    def get_body_comvel(self, body_name):
        #TODO : Returns the Center Of Mass velocity of a certain body
        # idx = self.model.body_names.index(six.b(body_name))
        # return self.model.body_comvels[idx]
        pass

    def get_body_xmat(self, body_name):
        # TODO:Next what is this thingi ? rotation matrix ?
        # idx = self.model.body_names.index(six.b(body_name))
        # return self.model.data.xmat[idx].reshape((3, 3))
        pass
