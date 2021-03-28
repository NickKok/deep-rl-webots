"""supervisor controller."""

import sys
import os
sys.path.append(os.path.abspath("__GLOBALPATH__/biorob-rl-communication-lib"))

from webotsConnection import SimpleWebotsPeripheralSystem


SOLIDS = [
'MODULE_1',
'MODULE_2',
'MODULE_3',
'MODULE_4',
'MODULE_5',
'MODULE_6',
'MODULE_7',
'MODULE_8'
]


myBrain = SimpleWebotsPeripheralSystem(SOLIDS,"emitter")


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while myBrain.step() != -1:
    try:
        # We get the signals from the motor cortex and send them to the spinal cord
        aa = myBrain.getFromMotorCortex() - 0.5
        #print(aa)
        myBrain.sendToSpinalCord(aa)
        # We receive signals from sensory cortex and send that to Cortex
        myBrain.sendToPremotorCortex(myBrain.receiveFromSensoryCortex())
    except:
        print("I will restart soon")
        myBrain.supervisor.simulationRevert()
