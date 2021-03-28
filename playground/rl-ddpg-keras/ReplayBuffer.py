"""
Data structure for implementing experience replay

Author: Florin Dzeladini
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size,random_seed):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.terminals = deque()
        random.seed(random_seed)

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.count < batch_size:
            return random.sample(self.buffer, self.count)
        else:
            return random.sample(self.buffer, batch_size)

    def getBatchTerminals(self, batch_size):
        # Get one random terminals
        terminalPoss = np.array(self.terminals).nonzero()[0]
        if len(terminalPoss) == 0:
            #print "Still no terminals, normal batching"
            return self.getBatch(batch_size)
        else:
            terminalPos = random.sample(list(terminalPoss),1)[0]
            if(terminalPos < batch_size):
                return np.array(self.buffer)[0:terminalPos]
            else:
                return np.array(self.buffer)[terminalPos-batch_size:terminalPos]

    def size(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.count

    def add(self, state, action, reward, new_state, terminal):
        experience = (state, action, reward, new_state, terminal)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            self.terminals.popleft()
        if(terminal):
            self.terminals.append(1)
        else:
            self.terminals.append(0)

    def erase(self):
        self.buffer = deque()
        self.count = 0
