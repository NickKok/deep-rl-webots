#!/usr/bin/env python

import sys, time




# Webots RPC

import pyRpc
import numpy as np
import logging
#logging.basicConfig(level=logging.DEBUG)



class WebotsRPC:
	'''
	This class allows to extend the capabilities of
	a controller and allow it to be controlled by an external sources
	'''
	actions = []
	observations = []
	server = None

	workersNumber = 0
	workerStatus = [] # 0 = 1 = acting

	def __init__(self,actionsInit,observationsInit,workersNumber):
		self.actions = actionsInit
		self.observations = observationsInit
		self.workersNumber = workersNumber
		# using local IPC communication
		self.server = pyRpc.PyRpc("Server", workers=self.workersNumber)
		self.workerStatus = np.zeros(self.workersNumber)
		# Could have used a TCP connection if we wanted:
		# server = pyRpc.PyRpc("Server", tcpaddr="127.0.0.1:40000")

	def start(self):
		self.server.start()

	def stop(self):
		self.server.stop()



	def waitForWorkersActionRequest(self):
		# Send actions to workers
		if(sum(self.workerStatus) == self.workersNumber):
			return False
		else:
			return True

	def waitForWorkersObservation(self):
		# Send actions to workers
		if(sum(self.workerStatus) == 0):
			return False
		else:
			return True


GET_ACT = 0
SET_OBS = 1



def main():
	actionsInit = np.random.randn(10,1)
	observationsInit = np.random.randn(10,1)
	workersNumber = 1
	workersStatus = np.zeros(workersNumber)
	server = WebotsRPC(actionsInit,observationsInit,workersNumber)

	def step(actions):
		print actions
		server.actions = actions
		server.observations = np.random.randn(10,1)
		server.workersSentActions += 1
		return server.observations;

	def setObs(obs,*arg):
		if(server.workerStatus[arg[0]] == SET_OBS): # If the worker is in the right status
			# Its either one observation per worker if more than one worker or a set of observations
			if(len(arg) == 1): # we set the observation of a specific worker
				server.workerStatus[arg[0]] = GET_ACT
				server.observations[arg[0]] = obs
				return 1
			elif(len(arg) == 0):
				server.workerStatus[0] = GET_ACT
				server.observations = obs
				return 1
		else:
			print "#Error observation already sent."
			return -1

	def getAct(*arg):
		if(server.workerStatus[arg[0]] == GET_ACT): # If the worker is in the right status
			# Its either one observation per worker if more than one worker or a set of observations
			if(len(arg) == 1): # we set the observation of a specific worker
				server.workerStatus[arg[0]] = SET_OBS
				return server.actions[arg[0]]
			elif(len(arg) == 0): # default mode only one worker.
				server.workerStatus[0] = SET_OBS
				return server.actions
		else:
			print "#Error action already retrieved."
			return -1


	server.server.publishService(setObs)
	server.server.publishService(getAct)
	server.start()

	try:
		# Run one controller step
		while True:
			server.actions = -1*np.random.randn(10,1)
			#print "Waiting for workers to retrieve action"
			# Wati for workers actions request
			while server.waitForWorkersActionRequest():
				pass
			#print "All actions retrieved.. Worker is acting"
			while server.waitForWorkersObservation():
				pass
			#print "All observation retrieved..", server.observations
	except KeyboardInterrupt:
		server.stop()



if __name__ == "__main__":
	main()
