'''
ReplayManager class manages replay memory.
'''

import random

class ReplayManager:
    def __init__(self, replay_cap):
        self.capacity = replay_cap
        self.memory = []
        self.counter = 0

    def add_mem(self, experience):
        # if below memory cap, just fill replay memory
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else: # otherwise, replace a memory
            self.memory[self.counter % self.capacity] = experience

        self.counter += 1

    '''
    @param: batch_size number of memories to sample (NN hyperparam)
    '''
    def sample_batch(self, batch_size):
        if len(self.memory) > batch_size:
            return random.sample(self.memory, batch_size)
        else:
            raise ValueError("Too few memories to sample")