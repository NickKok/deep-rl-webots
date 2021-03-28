"""brainConnector_bidirectional controller."""

import sys
import os
sys.path.append(os.path.abspath("__GLOBALPATH__/biorob-rl-communication-lib"))

from webotsConnection import SimpleWebotsPeripheralSystem


SOLIDS = [
    "F2_SOLID",
    "F1_SOLID",
    "F1_FRONT_LEFT_1_SOLID",
    "AX12_FRONT_LEFT_2_SOLID",
    "F2_NECK1_SOLID",
    "AX12_NECK_2_SOLID",
    "F1_FRONT_RIGHT_1_SOLID",
    "AX12_FRONT_RIGHT_2_SOLID",
    "PELVIS_1_SOLID",
    "F1_BACK_LEFT_1_SOLID",
    "AX12_BACK_LEFT_2_SOLID",
    "F1_BACK_RIGHT_1_SOLID",
    "AX12_BACK_RIGHT_2_SOLID"
]


myBrain = SimpleWebotsPeripheralSystem(SOLIDS,"emitter")


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while myBrain.step() != -1:
#    try:
        # We get the signals from the motor cortex and send them to the spinal cord
        myBrain.sendToSpinalCord(myBrain.getFromMotorCortex())
        # We receive signals from sensory cortex and send that to Cortex
        myBrain.sendToPremotorCortex(myBrain.receiveFromSensoryCortex())
#    except:
#        print "I will restart soon"
