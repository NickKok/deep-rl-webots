"""
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.terminals = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)

        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            self.terminals.popleft()
        if(t):
            self.terminals.append(1)
        else:
            self.terminals.append(0)

    def size(self):
        return self.count

    def sample_batch_terminals(self, batch_size):
        # Get one random terminals
        terminalPoss = np.array(self.terminals).nonzero()[0]
        if len(terminalPoss) == 0:
            #print "Still no terminals, normal batching"
            return self.sample_batch(batch_size)
        else:
            terminalPos = random.sample(terminalPoss,1)[0]
            batch = []
            if(terminalPos < batch_size):
                batch = np.array(self.buffer)[0:terminalPos]
            else:
                batch = np.array(self.buffer)[terminalPos-batch_size:terminalPos]
            s_batch = np.array([_[0] for _ in batch])
            a_batch = np.array([_[1] for _ in batch])
            r_batch = np.array([_[2] for _ in batch])
            t_batch = np.array([_[3] for _ in batch])
            s2_batch = np.array([_[4] for _ in batch])
            return s_batch, a_batch, r_batch, t_batch, s2_batch

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
