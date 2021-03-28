"""
Gym Environment for the Regis robot of the
Webots simulator (www.cyberbotics)

see https://bitbucket.org/efx/libnmm for details

Author: Florin Dzeladini
"""
import numpy as np
from gym import utils
from gym.envs.webots import webots_env
import matplotlib.pyplot as plt
import pprint as pp

class STATES:
    SWING = 0
    STANCE = 1


class RegisBrain1(webots_env.WebotsEnv):
    def __init__(self, action_dim = 11, repeat = 4, port = 5562, world = ""):
        self.done = False
        self.rews = None
        self.lastLeftStateChange = -1.0
        self.lastRightStateChange = -1.0
        self.minimumStateDuration = 0.3
        self.leftState = STATES.SWING
        self.rightState = STATES.SWING

        if repeat < 0:
            self.updateType = 1 # Until touchdown

        self.state = dict(
            current = dict(
                #lastExtraInfo = None,
                stepNumber = 0,
                time = 0,
            )
        )
        self.action_dim = action_dim
        REGIS_CFG = {
            'obs_dim' : None,
            'action_space' : np.zeros([action_dim,2])+np.array([0,1])
        }
        if(world == ""):
            raise ValueError("world can not be empty")
        webots_env.WebotsEnv.__init__(self, REGIS_CFG, repeat, port, world)

    def _step(self,action):
        ob = []
        reward = 0

        while True:

            if self.model.dataThere:
                self.updateState()
                reward = self.getReward()
                self.done = self.isDone()
            self.do_one_simulation_step(action)
            ob = self.state_vector()
            if self.done or ob[-3] == 1.0 or ob[-2] == 1.0:
                break

        # if self.model.dataThere:
        #     self.updateState()
        #     reward = self.getReward()
        #     self.done = self.isDone()
        # self.do_simulation(action, self.frame_skip)
        # ob = self.state_vector()

        if(self.done):
            # Reward overwritten by fitness.
            # Distance
            reward = self.getCoM(self.model.data['qpos'])[2]
            # Speed
            #distance=np.linalg.norm(self.model.data['qpos'][0][:])
            #duration=self.state["current"]["time"]
            #reward = distance/10 if distance < 10 else duration-(distance/duration-0.4)**2 # Reward speed
            self.finalize()

        return ob, reward, self.done, dict()

    def updateState(self):
        self.state["current"]["time"] += self.frame_skip*self.model.opt['timestep']
        self.state["current"]["lastExtraInfo"] = self.model.data['extrainfo']

    def finalize(self):
        self.state["current"]["stepNumber"] = 0.0
        self.state["current"]["time"] = 0.0
        self.leftState = STATES.SWING
        self.rightState = STATES.SWING
        self.lastLeftStateChange = -1.0
        self.lastRightStateChange = -1.0


    def getCoM(self,q=None):
        if q is None:
            q = self.model.data['qpos']
        m = np.array(self.model.data['mass'])
        x = (q[:,0].dot(m))/sum(m)
        y = (q[:,1].dot(m))/sum(m)
        z = (q[:,2].dot(m))/sum(m)
        return np.array([x,y,z])

    def isDone(self):
        CoM = self.getCoM(self.model.data['qpos'])
        com_y = CoM[1]
        done = bool((com_y-self.model.data['qpos'][-1][1] < 0.8))
        return done;

    def state_vector(self):
        left_touch_down = 0
        right_touch_down = 0
        if("lastExtraInfo" in self.state["current"]):
            threshold = 0.01
            left_foot = self.model.data['extrainfo'][10]
            left_foot_prev = self.state["current"]["lastExtraInfo"][10]
            right_foot = self.model.data['extrainfo'][11]
            right_foot_prev = self.state["current"]["lastExtraInfo"][11]
            if( left_foot_prev < threshold and
                left_foot >= threshold and
                self.state["current"]["time"]-self.lastLeftStateChange > self.minimumStateDuration and
                self.leftState != STATES.STANCE
                ):
                self.leftState = STATES.STANCE
                left_touch_down = 1
                self.lastLeftStateChange = self.state["current"]["time"]

            if( left_foot_prev > threshold and
                left_foot <= threshold and
                self.state["current"]["time"]-self.lastLeftStateChange > self.minimumStateDuration and
                self.leftState != STATES.SWING
                ):
                self.leftState = STATES.SWING
                self.lastLeftStateChange = self.state["current"]["time"]

            if( right_foot_prev < threshold and
                right_foot >= threshold and
                self.state["current"]["time"]-self.lastRightStateChange > self.minimumStateDuration and
                self.rightState != STATES.STANCE
                ):
                self.rightState = STATES.STANCE
                right_touch_down = 1
                self.lastRightStateChange = self.state["current"]["time"]

            if( right_foot_prev > threshold and
                right_foot <= threshold and
                self.state["current"]["time"]-self.lastRightStateChange > self.minimumStateDuration and
                self.rightState != STATES.SWING
                ):
                self.rightState = STATES.SWING
                self.lastRightStateChange = self.state["current"]["time"]
            #if(left_touch_down+right_touch_down > 0):
            #    print("{} {}".format(left_touch_down,right_touch_down))
        #print("{}".format(self.model.data['extrainfo']))
        return np.concatenate([
            self.model.data['extrainfo'][0:9],
            [left_touch_down],
            [right_touch_down],
            [self.getCoM(self.model.data['qpos'])[2]]
        ])


    def getReward(self):
        # We want the reward to be state dependent.
        #

        reward = 0.0
        OPTIMIZE_FOR = "faster than my ghost"
        if(OPTIMIZE_FOR == "faster than my ghost"):
            v = 0.5 # [m/s]
            t = self.state["current"]["time"] # [s]
            X = self.getCoM(self.model.data['qpos']) # [m,m,m]
            x = X[2] # position
            y = X[1]
            rewardx = x-v*t
            rewardy = y-1.0
            reward = -(rewardx*rewardx)
            reward = min(1,(max(-1,reward)))
        else:
            print("Something is definitely wrong, you try to optimizer for '{}' but this does not exist".format(OPTIMIZE_FOR))

        return reward
