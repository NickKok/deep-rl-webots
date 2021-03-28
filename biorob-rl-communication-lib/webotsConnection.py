# Simple supervisor that handles the communication with the rpc server

from controller import Supervisor as Robot

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


class SimpleWebotsPeripheralSystem():

    def __init__(self,SOLIDS,emitterName,*args):
        if(type(SOLIDS[0])==tuple):
            self.SOLIDS = [i for (i,v) in SOLIDS]
            self.MASSES = [v for (i,v) in SOLIDS]
        else:
            self.SOLIDS = SOLIDS
            self.MASSES = np.zeros(len(self.SOLIDS))
        self.supervisor = Robot()
        self.fast_mode = True
        self.modeSwitched = False
        self.time = 0
        self.timestep = int(self.supervisor.getBasicTimeStep())
        # Receiver is not mandatory, view it as local information from your spinal cord
        # that you want to send back to the brain
        self.receiver = None
        if len(args) > 1:
            self.client = SimpleRPCClient(args[1])
        else:
            self.client = SimpleRPCClient()
        if len(args) > 0:
            self.receiver = self.supervisor.getReceiver(args[0])
            if self.receiver != None:
                self.receiver.enable(self.timestep)
        # Emitter is mandatory, view it as the local information your spinal cord
        # needs
        self.emitter = self.supervisor.getEmitter(emitterName)


    def sendToPremotorCortex(self,obs):
        self.client.send(obs)

    def sendToSpinalCord(self,actions,**kwargs):
        if 'encrypt_with' in kwargs:
            if kwargs['encrypt_with'] == 'json':
                self.emitter.send(str.encode(json.dumps(actions)))
                return
        self.emitter.send(pickle.dumps(actions, protocol=0))
        return

    def receiveFromSensoryCortex(self):
        extraInfo = []
        if self.receiver != None:
            while self.receiver.getQueueLength() > 0:
                extraInfo = json.loads(self.receiver.getData())
                # do something with the map
                self.receiver.nextPacket()
        return {
            'extrainfo' : extraInfo,
            'qpos' : self.getPosition(),
            'qvel' : self.getVelocity(),
            'com' : self.getCenterOfMass(),
            'mass' : self.getMasses(),
            'phase' : np.mod(2.0 * pi * 1.0 * self.time,2.0 * pi), # TODO put real phase,
            'time' : self.time,
            'timestep' : int(self.time/(self.timestep/1000.0))
        }
    def getFromMotorCortex(self):
        msg = self.client.get()
        if "fast_mode" in msg:
            if msg["fast_mode"] != self.fast_mode or not self.modeSwitched:
                self.modeSwitched = True
                self.fast_mode = msg["fast_mode"]
                print("Fast mode switched to {} by an external source".format(self.fast_mode))
                if(self.fast_mode):
                    self.supervisor.simulationSetMode(3)
                else:
                    self.supervisor.simulationSetMode(1)
        else:
            print("No fast_mode in msg from rl")
        if "reset_model" in msg :
            if msg["reset_model"] == True:
                self.supervisor.simulationQuit(1)
                time.sleep(10.0)
        else:
            print("No reset_model in msg from rl")
        if "act" in msg:
            return msg["act"]
        else:
            print("No action in msg from rl")
            return np.zeros(10)

    def getPosition(self):
        # for m in self.SOLIDS:
        #     print "{}:{}".format(m,self.supervisor.getFromDef(m))
        return([
          self.supervisor.getFromDef(m).getPosition() for m in self.SOLIDS
          ])

    def getVelocity(self):
       return([
          self.supervisor.getFromDef(m).getVelocity() for m in self.SOLIDS
          ])

    def getCenterOfMass(self):
       return([
          self.supervisor.getFromDef(m).getCenterOfMass() for m in self.SOLIDS
          ])

    def getMasses(self):
       return(self.MASSES)

    def step(self):
        self.time += self.timestep/1000.0
        return self.supervisor.step(self.timestep)
