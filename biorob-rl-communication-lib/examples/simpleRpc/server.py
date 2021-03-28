#
#  Synchronized action
#
import zmq
import numpy as np
import pickle

import sys
import os
sys.path.append(os.path.abspath("/home/efx/Development/PHD/RL/deep-rl-biorob/webots_communication_toolbox/tools"))

try:
    from simplerpc import WebotsCommunicatorService
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install Webots_py, and also perform the setup instructions here: https://gitlab.com/srill-fb99/deep-rl-biorob.)".format(e))



def main():
    server = WebotsCommunicatorService(7,np.array([
                            [ 0,  1],
                            [ 0,  1]
                                ]))

    # Now broadcast exactly 1M updates followed by END
    reset_model = False
    for i in range(10000):
        if ( i==1000 ):
            reset_model = True

        act = np.asarray(np.random.randn(8,1))
        server.step(act)

        if(reset_model == True):
            reset_model = False
        #print "rec:obs {}:{}".format(i,obs[2][0])

    server.syncservice.send(b'END')

if __name__ == '__main__':
    main()
