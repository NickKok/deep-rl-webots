"""
Gym Environment for the Yamor robot of the
Webots simulator (www.cyberbotics)

Author: Florin Dzeladini
"""
import numpy as np
from gym import utils
from gym.envs.webots import webots_env

YAMOR_CFG = {
    'obs_dim' : 7,
    'action_space' : np.array([
        [ -1,  1],
        [ -1,  1],
        [ -1,  1],
        [ -1,  1],
        [ -1,  1],
        [ -1,  1],
        [ -1,  1],
        [ -1,  1]
    ])
}

class Yamor(webots_env.WebotsEnv):

    time = 0
    doneTimes = 0
    hardStopLongitudinalConstrain = -0.05



    def __init__(self):
        webots_env.WebotsEnv.__init__(self, YAMOR_CFG, 2)


    def _step(self,action):
        self.time += self.frame_skip*self.model.opt['timestep']
        reward_forward = 0
        reward_leg = 0
        reward_survive = 0

        # to use velocities for reward : maximizes acceleration direction
        optimizeFor = 'qvel'
        # to use positions for reward : maximizes speed direction
        #optimizeFor = 'qpos'

        if(len(self.model.data[optimizeFor]) == 0): # if first step
            self.do_simulation(action, self.frame_skip)
        else:
            # 1) Save state before doing action
            head_vec_before = self.model.data[optimizeFor][0,0:3]
            mid_vec_before = self.model.data[optimizeFor][3,0:3]
            leg_vec_before = 0.5*(self.model.data[optimizeFor][0,0:3]+self.model.data[optimizeFor][-1,0:3])
            target_vec_longitudinal_motion = [0,0,-1]
            target_vec_armlike_motion = np.array([1,0,0])
            target_vec_armlike_motion = target_vec_armlike_motion/np.linalg.norm(target_vec_armlike_motion)
            # 2) Do action
            self.do_simulation(action, self.frame_skip)
            head_vec_after = self.model.data[optimizeFor][0,0:3]
            leg_vec_after = 0.5*(self.model.data[optimizeFor][0,0:3]+self.model.data[optimizeFor][-1,0:3])
            mid_vec_after = self.model.data[optimizeFor][3,0:3]
            # 3) Compute reward
            #reward = np.dot(target_vec,(head_vec_after-head_vec_before))
            ndiffHead = mid_vec_after-mid_vec_before
            norm = np.linalg.norm(ndiffHead)
            if norm != 0:
                ndiffHead = ndiffHead/norm
            ndiffLeg = leg_vec_after-leg_vec_before
            norm = np.linalg.norm(ndiffLeg)
            if norm != 0:
                ndiffLeg = ndiffLeg/norm
            reward_forward = np.dot(target_vec_armlike_motion,ndiffHead)
            reward_leg = np.dot(target_vec_armlike_motion,ndiffLeg)


        # 4) Return everything
        ob = self.state_vector()

        done = False

        # hardStopLongitudinalConstrain IS
        # = self.model.data['qpos'][4][0]-0.05
        # = -0.05 otherwise
        if(self.model.data['qpos'][4][0] > -0.05 and self.model.data['qpos'][4][0] - 0.05 > self.hardStopLongitudinalConstrain):
            self.hardStopLongitudinalConstrain = self.model.data['qpos'][4][0] - 0.05




        if(self.time > 10+self.doneTimes/30.0):
            self.time = 0;
            done = True
        #if((self.time > 1.0 and self.model.data['qpos'][0][2] > 0) ):
        if(self.model.data['qpos'][4][0] < 0.0):
            reward_survive = -10
        if((self.time > 1.0 and self.model.data['qpos'][4][0] < -0.05) ):
            reward_survive = -10
            self.time = 0;
            done = True

        if(done):
            self.hardStopLongitudinalConstrain = -0.05
            print("done, next will last {} sec.".format(10+self.doneTimes/30.0))
            self.doneTimes += 1


        reward = max(0,reward_forward)+max(0,reward_leg)+reward_survive
        return ob, reward, done, dict(
            reward_forward=max(0,reward_forward),
            reward_ctrl=max(0,reward_leg),
            reward_contact=0,
            reward_survive=reward_survive)
    def state_vector(self):
        # State vector not in absolute coordinate but relative to the
        # average position
        #ref = sum(self.model.data['qpos'][:,:])/len(self.model.data['qpos'])
        ref = self.model.data['qpos'][3,:] # only a subpart used for ref
        return np.concatenate([
            np.array([self.model.data['phase']]).flat,
            (self.model.data['qpos'][:,:]-ref).flat
            #https://www.cyberbotics.com/doc/reference/supervisor#wb_supervisor_node_get_velocity
            #TODO : not clear how to transform angular velocities in local coordinate
            #(self.model.data['qvel'][4:6,3:]-ref).flat # velocities is a 6 DOF in webots
        ])
