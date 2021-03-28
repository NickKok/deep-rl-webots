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


ASYNC_CALLS = 0

def callback(resp, *args, **kwargs):
	global ASYNC_CALLS
	# print "Got fast response:", resp.result
	ASYNC_CALLS += 1

if __name__ == "__main__":

	remote = RpcConnection("Server", workers=1)
	# if the server were using a TCP connection:
	# remote = RpcConnection("Server", tcpaddr="127.0.0.1:40000")

	# time.sleep(.1)

	print "Waiting on async calls to finish"
	while ASYNC_CALLS < 1000:
		actions = np.random.randn(10,1)
		observations = remote.call("step", async=True, callback=callback,args=([np.asarray(actions)]))
		pass
