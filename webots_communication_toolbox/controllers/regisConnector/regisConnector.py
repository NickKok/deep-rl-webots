"""supervisor controller."""

import sys
import os
import json
sys.path.append(os.path.abspath("__GLOBALPATH__/biorob-rl-communication-lib"))
import argparse
import pprint as pp
from webotsConnection import SimpleWebotsPeripheralSystem
from simplerpc import SomebodyIsAlreadyConnected
import numpy as np



SOLIDS = [
 ('REGIS', 53.5),
 ('LEFT_THIGH', 8.5),
 ('LEFT_SHIN', 3.5),
 ('LEFT_FOOT', 1.25),
 ('RIGHT_THIGH', 8.5),
 ('RIGHT_SHIN', 3.5),
 ('RIGHT_FOOT', 1.25)
]


parser = argparse.ArgumentParser(description='provide arguments for your RL(A3C) agent')
parser.add_argument('--run-type', help='type of run : 0 (learn) or 1 (normalize) default 0', default=0) # 1e-4
parser.add_argument('--random-seed-port', help='port and random seed to used for this worker', default=0) # 1e-4
args = vars(parser.parse_args())
#pp.pprint(args)


np.random.seed(int(args["random_seed_port"]))
currentPort = int(args["random_seed_port"])
run_type = int(args["run_type"])



def float_to_str(f):
    float_string = repr(f)
    if 'e' in float_string:  # detect scientific notation
        digits, exp = float_string.split('e')
        digits = digits.replace('.', '').replace('-', '')
        exp = int(exp)
        zero_padding = '0' * (abs(int(exp)) - 1)  # minus 1 for decimal point in the sci notation
        sign = '-' if f < 0 else ''
        if exp > 0:
            float_string = '{}{}{}.0'.format(sign, digits, zero_padding)
        else:
            float_string = '{}0.{}{}'.format(sign, zero_padding, digits)
    if(float_string == "-0.0" or float_string == "0.0"):
        return "0"
    return float_string
    
myBrain = SimpleWebotsPeripheralSystem(SOLIDS,"emitter","receiver",currentPort)




# Preparation steps (we run without applying any change)
#for i in range(2500):
#    myBrain.step()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while myBrain.step() != -1:
    #try:
        # We get the signals from the motor cortex and send them to the spinal cord
        actions = myBrain.getFromMotorCortex()
        actions = [float_to_str(a) for a in actions]
        myBrain.sendToSpinalCord(actions, encrypt_with = 'json')
        # We receive signals from sensory cortex and send that to Cortex
        obs = myBrain.receiveFromSensoryCortex()
        #print(obs['extrainfo'])
        myBrain.sendToPremotorCortex(obs)
    #except Exception as e:
    #    print(e)
    #    #myBrain.supervisor.simulationQuit(1)



