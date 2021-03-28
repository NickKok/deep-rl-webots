import sys
import os
import json
sys.path.append(os.path.abspath("__GLOBALPATH__/biorob-rl-communication-lib"))

from genericConnection import GenericPeripheralSystem
import numpy as np

myBrain = GenericPeripheralSystem()

aa = np.zeros([23,1])

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while myBrain.step() != -1:
    aa = myBrain.getFromMotorCortex()
    myBrain.sendToSpinalCord(aa.tolist(), encrypt_with = 'json')
    #print(aa)
    bb = myBrain.receiveFromSensoryCortex()
    #print(bb)
    myBrain.sendToPremotorCortex(bb)
