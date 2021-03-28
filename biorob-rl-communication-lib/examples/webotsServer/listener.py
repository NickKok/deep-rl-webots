#!/usr/bin/env python

import sys, time
import pyRpc

import logging

#logging.basicConfig(level=logging.DEBUG)


def main():
    # using local IPC communication
    server = pyRpc.PyRpc("Server", workers=2)
    # Could have used a TCP connection if we wanted:
    # server = pyRpc.PyRpc("Server", tcpaddr="127.0.0.1:40000")

    server.publishService(step)
    server.publishService(noReturn)
    server.start()


    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.stop()

counter = 0


def step(actions):
    print actions
    obs = np.random.randn(10,1)
    return obs;

def noReturn(value=1):
    "This does something and returns nothing"
    print "noReturn() called!"
    time.sleep(2)
    print "noReturn() done!"
    return 1


if __name__ == "__main__":
    main()
