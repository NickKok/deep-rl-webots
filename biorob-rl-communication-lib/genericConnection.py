import time
# For serialization
#import cPickle as pickle
import pickle
import json

import numpy as np
from math import pi,sin

import sys
import os
sys.path.append(os.path.abspath("__GLOBALPATH__/biorob-rl-communication-lib"))

from simplerpc import SimpleRPCClient, DumpBrainConnection

class GenericPeripheralSystem():

    def __init__(self,timestep=1):
        self.client = SimpleRPCClient()
        self.time = 0
        self.timestep = timestep
        self.fast_mode = False

    def step(self):
        self.time += self.timestep/1000.0
        return 1

    def sendToSpinalCord(self,actions,**kwargs):
        pass

    def receiveFromSensoryCortex(self):
        extraInfo = [0,0,0,0]
        return {
            'extrainfo' : extraInfo,
            'qpos' : 10+ np.zeros([50,3]),
            'qvel' : 10+ np.zeros([50,6]),
            'com' :  10+ np.zeros([50,3]),
            'phase' : 0.0,
            'timestep' : 1.0/1000.0
        }

    def sendToPremotorCortex(self,obs):
        self.client.send(obs)

    def getFromMotorCortex(self):
        msg = self.client.get()
        if "fast_mode" in msg:
            if msg["fast_mode"] != self.fast_mode:
                self.fast_mode = msg["fast_mode"]
                print("Fast mode switched to {} by an external source".format(self.fast_mode))
        else:
            print("No fast_mode in msg from rl")
        if "reset_model" in msg :
            if msg["reset_model"] == True:
                self.client.sync()
        else:
            print("No reset_model in msg from rl")
        if "act" in msg:
            return msg["act"]
        else:
            print("No action in msg from rl")
            return np.zeros(10)
