
# IMPORTS
from __future__ import absolute_import, division, print_function

import random

import tensorflow as tf
import numpy as np
import math


class DQN(tf.keras.Model):
    def __init__(self, memory_size, layers):
        super(DQN, self).__init__()
        self.memory_size = memory_size
        self.layer_params = layers

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(memory_size,))

        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layers]

        # DQN should output 441 action possibilities. This represents the continuous action space discretized.
        # There are 21 steering options (-1.0 to 1.0 at 0.1 increments) and 21 angle options. Conversion from this
        # flattened representation is performed automatically in the python environment
        self.output_layer = tf.keras.layers.Dense(441, activation='linear')

    '''
    Overriding feed-forward method for keras models
    '''
    @tf.function
    def call(self, inputs, **kwargs):
        input = self.input_layer(inputs)
        for layer in self.hidden_layers:
            input = layer(input)

        output = self.output_layer(input)
        return output

class DQNAgent:
    def __init__(self, e_decay=0.001):
        self.current_step = 0
        self.e_decay = e_decay

    # helper function to derive the current exploration rate of our e-greedy strategy
    def strategy(self):
        return math.exp(-1*self.current_step*self.e_decay)

    def action(self, state, dqn_net):
        # get the probability of selecting a random action instead of e-greedy from policy class
        rate = self.strategy()
        self.current_step += 1

        if rate > random.random():
            # return random action, exploration rate at current step, indicator that action was random
            return random.randrange(441), rate, False
        else:
            # return argmax_a q(s, a), rate at current step, indicator that action was greedy selected
            return np.argmax(dqn_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), rate, True

