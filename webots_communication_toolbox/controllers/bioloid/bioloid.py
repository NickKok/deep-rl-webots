"""supervisor controller."""

# Simple supervisor that handles the communication with the rpc server

from controller import Supervisor

import time
import pickle  # to serialize the map

import numpy as np
from math import pi,sin

import sys
import os

supervisor = Supervisor()

timestep = int(supervisor.getBasicTimeStep())

moduleNames = [
    "pelvis",
    "front_left_1",
    "front_right_1",
    "front_left_2",
    "front_right_2",
    "front_left_3",
    "front_right_3",
    "back_left_1",
    "back_right_1",
    "back_left_2",
    "back_right_2",
    "back_left_3",
    "back_right_3",
    "head",
]

for m in moduleNames:
    supervisor.getPositionSensor(m+"_sensor").enable(timestep)


def act(motorName,action):
    shift = action;  # phase shift for this module
    if(type(shift) == np.ndarray):
        shift = shift[0]
    phase = 2.0 * pi * F * time;

    factor = 1.0
    if factor*shift > 1.5:
        shift = 1.5*factor;
    if factor*shift < -1.5:
        shift = -1.5*factor;
    #robot.getMotor("motor").setPosition(factor*shift)
    #supervisor.getMotor(motorName).setPosition(A * sin(phase+shift))
    supervisor.getMotor(motorName).setPosition(A * shift)


time = 0


F = 0.5;            # frequency
A = 3.14;            # amplitude

receiver = supervisor.getReceiver("receiver")
receiver.enable(timestep)


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while supervisor.step(timestep) != -1:
    time += timestep/1000.0
    
    while receiver.getQueueLength() > 0:
      actions = pickle.loads(receiver.getData())
      # do something with the map
      receiver.nextPacket()
      for i,motor in enumerate(moduleNames):
        act(motor,actions[i])
  


# Enter here exit cleanup code.
