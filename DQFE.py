import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image
import reverb
from collections import namedtuple
from ReplayManager import ReplayManager

from DQN import DQN

class DQFE(DQN):
    def __init__(self, memory_size, layers, replay_cap):
        super().__init__(memory_size, layers)
        self.layer_params = layers
        self.memory_manager = ReplayManager(replay_cap)

    def pre_train(self, batch_size, gamma, optimizer, t_expd, pre_train_rounds, trained_agent, env, dqnagent, update_freq):
        target_policy = DQFE(self.memory_size, self.layer_params, self.memory_manager)

        # ensure equivalence of policy network and target policy network at start
        policy_params = self.trainable_variables
        optimal_params = target_policy.trainable_variables
        for pvar, ovar in zip(policy_params, optimal_params):
            ovar.assign(pvar.numpy())

        Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'states_prime', 'terminates'])

        pro_memory_manager = ReplayManager(self.memory_manager.capacity)

        # collect 3 rounds of experience from our professional driver
        for i in range(3):
            state = env.reset(relaunch=True)

            terminated = False
            while not terminated:
                # collect "Professional" agent action selection
                action = np.argmax(trained_agent(np.atleast_2d(np.atleast_2d(state).astype('float32'))))

                # apply action
                state_prime, reward, terminated = env.step(action)

                # store experience in replay buffers
                self.memory_manager.add_mem(Experience(state, action, reward, state_prime, terminated))
                pro_memory_manager.add_mem(Experience(state, action, reward, state_prime, terminated))

                # s <- s'
                state = state_prime

                if terminated:
                    break

        # train normally for pre_train_rounds
        for episode in range(pre_train_rounds):
            curr_state = env.reset()
            episode_step_count = 0
            terminated = False

            while not terminated:
                episode_step_count += 1

                # select action according to e-greedy policy
                action, rate, greedy = dqnagent.action(curr_state, self)
                state_prime, reward, terminated = env.step(action)
                # store experience in replay buffer
                self.memory_manager.add_mem(Experience(curr_state, action, reward, state_prime, terminated))

                # s <- s'
                curr_state = state_prime

                if len(self.memory_manager.memory) > batch_size:
                    # collect replay sample
                    memories = self.memory_manager.sample_batch(batch_size)
                    minibatch = Experience(*zip(*memories))

                    # convert batch to numpy array for tensorflow compatibility
                    statelist = np.asarray(minibatch[0])
                    actionlist = np.asarray(minibatch[1])
                    rewardlist = np.asarray(minibatch[2])
                    state_primes = np.asarray(minibatch[3])
                    terminates = np.asarray(minibatch[4])

                    # calculate loss and apply gradient descent
                    q_prime = np.max(target_policy(np.atleast_2d(state_primes).astype('float32')), axis=1)
                    q_optimal = np.where(terminates, rewardlist, rewardlist + gamma * q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        q = tf.math.reduce_sum(
                            self(np.atleast_2d(statelist).astype('float32')) * tf.one_hot(actionlist, 441),
                            axis=1)
                        loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                    # Update the policy network weights using ADAM
                    variables = self.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))

                # to prevent the proverbial 'dog chasing its own tail' of updating the policy network every time, we only update
                # the target network every update_freq times
                if episode_step_count % update_freq == 0:
                    policy_params = self.trainable_variables
                    optimal_params = target_policy.trainable_variables

                    for pvar, ovar in zip(policy_params, optimal_params):
                        ovar.assign(pvar.numpy())

                # new episode if a terminal state is reached
                if terminated:
                    break

        # train using only the pro_memory_manager for t_expd rounds
        for episode in range(t_expd):
            curr_state = env.reset()
            episode_step_count = 0
            terminated = False

            while not terminated:
                episode_step_count += 1

                # select action according to e-greedy policy
                action, rate, greedy = dqnagent.action(curr_state, self)
                state_prime, reward, terminated = env.step(action)
                # store experience in replay buffer
                self.memory_manager.add_mem(Experience(curr_state, action, reward, state_prime, terminated))

                # s <- s'
                curr_state = state_prime

                if len(pro_memory_manager.memory) > batch_size:
                    # collect replay sample
                    memories = pro_memory_manager.sample_batch(batch_size)
                    minibatch = Experience(*zip(*memories))

                    # convert batch to numpy array for tensorflow compatibility
                    statelist = np.asarray(minibatch[0])
                    actionlist = np.asarray(minibatch[1])
                    rewardlist = np.asarray(minibatch[2])
                    state_primes = np.asarray(minibatch[3])
                    terminates = np.asarray(minibatch[4])

                    # calculate loss and apply gradient descent
                    q_prime = np.max(target_policy(np.atleast_2d(state_primes).astype('float32')), axis=1)
                    q_optimal = np.where(terminates, rewardlist, rewardlist + gamma * q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        q = tf.math.reduce_sum(
                            self(np.atleast_2d(statelist).astype('float32')) * tf.one_hot(actionlist, 441),
                            axis=1)
                        loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                    # Update the policy network weights using ADAM
                    variables = self.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))

                # to prevent the proverbial 'dog chasing its own tail' of updating the policy network every time, we only update
                # the target network every update_freq times
                if episode_step_count % update_freq == 0:
                    policy_params = self.trainable_variables
                    optimal_params = target_policy.trainable_variables

                    for pvar, ovar in zip(policy_params, optimal_params):
                        ovar.assign(pvar.numpy())

                # new episode if a terminal state is reached
                if terminated:
                    break