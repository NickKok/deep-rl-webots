#
#  Synchronized subscriber
#
import time
import pickle
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("/home/efx/Development/PHD/RL/deep-rl-biorob/webots_communication_toolbox/tools"))

from simplerpc import SimpleRPCClient

def main():
    client = SimpleRPCClient()
    time.sleep(1)
    client.sync()



    # Third, get our updates and report how many we got
    nbr = 0
    while True:
        msg = client.syncclient.recv()
        if msg == b'END':
            print "END"
            break
        else:
            rec = pickle.loads(msg)
            print "rec:act {}:{}".format(nbr,rec['act'][2][0])
            obs = np.asarray(np.random.randn(10,1))
            serialized_obs = pickle.dumps(obs, protocol=0)
            client.syncclient.send(serialized_obs)
        nbr += 1

    print ('Received %d updates' % nbr)

if __name__ == '__main__':
    main()
