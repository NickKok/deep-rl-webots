#!/usr/bin/env python

"""
In this example, make sure to start the server first,
as this client will try and communicate immediately.
"""
from pyRpc import RpcConnection

import time
import numpy as np
import logging
# logging.basicConfig(level=logging.DEBUG)



waitForAction = True
waitForObservationAck = True

def getActions(resp, *args, **kwargs):
	global waitForAction
	if(resp.result != -1):
		#print "Got action:", resp.result
		waitForAction = False

def setObservations(resp, *args, **kwargs):
	global waitForObservationAck
	if(resp.result != -1):
		#print "Observation Set:"
		waitForObservationAck = False




if __name__ == "__main__":
	WORKER_ID = 0

	remote = RpcConnection("Server", workers=1)
	# if the server were using a TCP connection:
	# remote = RpcConnection("Server", tcpaddr="127.0.0.1:40000
	time.sleep(1)
	counter = 0

	while True:
		print "timestep: ", counter;
		finished = False
		while not finished:
			resp = remote.call("getAct",args=([WORKER_ID]))
			if(resp.result != -1):
				finished = True
		#print "I got the action, let me do some stuff"
		#time.sleep(0.01)
		#print "done, I am sending back the obs"
		finished = False
		while not finished:
			#print "hello"
			obs = 1+0*np.random.randn(10,1)
			resp = remote.call("setObs",args=([0.1,WORKER_ID]))
			#resp = remote.call("setObs",args=([obs]))
			if(resp.result != -1):
				finished = True
		#print "I got the obs ack, I can wait back for next action"
		counter+=1
