'''
Main program for DQN vs DQFE comparison

Bennet Montgomery
20074049
17blm1
'''

# IMPORTS
from __future__ import absolute_import, division, print_function

from DQN import DQN, DQNAgent
from DQFE import DQFE
from ReplayManager import ReplayManager

from gym_torcs import TorcsEnv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from collections import namedtuple

# CONSTANTS
vision = False
save_models = True # set to False to not save a model for evaluation
model_dir = 'models'

def save_model(model, modelname):
    os.system('mkdir -p ' + model_dir)
    model.save(model_dir + '/' + modelname)

def load_model(model_name):
    return tf.keras.models.load_model(model_dir + '/' + model_name)

def train(model_name, pre_trained=False, trained_agent=None):
    # HYPERPARAMETERS
    batch_size = 32
    gamma = 0.9
    update_freq = 25
    replay_cap = 8000
    episodes = 2000
    log_freq = 1
    learning_rate = 0.001
    optimizer = tf.optimizers.Adam(learning_rate)
    layer_params = [200, 110, 110]
    pre_train_episodes = 80
    t_expd = 10

    # start TORCS
    test_env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # instantiate agent for DQN action selection
    dqnagent = DQNAgent()

    # only pre-train if DQFE
    if not pre_trained:
        memory = ReplayManager(replay_cap)

        policy = DQN(len(test_env.observation_space.sample()), layer_params)
        optimal_policy = DQN(len(test_env.observation_space.sample()), layer_params)
    else:
        policy = DQFE(len(test_env.observation_space.sample()), layer_params, replay_cap)
        policy.pre_train(
            batch_size=batch_size,
            gamma=gamma,
            optimizer=optimizer,
            t_expd=t_expd,
            pre_train_rounds=pre_train_episodes,
            trained_agent=trained_agent,
            env=test_env,
            dqnagent=dqnagent,
            update_freq=update_freq
        )

        optimal_policy = DQFE(policy.memory_size, policy.layer_params, replay_cap)

        memory = policy.memory_manager

    Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'state_primes', 'terminates'])

    # ensure equivalence of policy network and target policy network at start
    policy_params = policy.trainable_variables
    optimal_params = optimal_policy.trainable_variables

    for pvar, ovar in zip(policy_params, optimal_params):
        ovar.assign(pvar.numpy())

    # tracking total reward per episode for optimization comparison
    reward_history = np.empty(episodes)

    for episode in range(episodes):
        if episode % 3 == 0:
            curr_state = test_env.reset(relaunch=True)
        else:
            curr_state = test_env.reset()

        episode_return = 0
        episode_step_count = 0
        loss_history = []
        terminated = False

        while not terminated:
            episode_step_count += 1

            # select action according to e-greedy policy
            action, rate, greedy = dqnagent.action(curr_state, policy)
            state_prime, reward, terminated = test_env.step(action)
            episode_return += reward

            # store experience in replay buffer
            memory.add_mem(Experience(curr_state, action, reward, state_prime, terminated))

            # s <- s'
            curr_state = state_prime

            if len(memory.memory) > batch_size:
                # collect replay sample
                memories = memory.sample_batch(batch_size)
                minibatch = Experience(*zip(*memories))

                # convert batch to numpy array for tensorflow compatibility
                statelist = np.asarray(minibatch[0])
                actionlist = np.asarray(minibatch[1])
                rewardlist = np.asarray(minibatch[2])
                state_primes = np.asarray(minibatch[3])
                terminates = np.asarray(minibatch[4])

                # calculate loss and apply gradient descent
                q_prime = np.max(optimal_policy(np.atleast_2d(state_primes).astype('float32')), axis=1)
                q_optimal = np.where(terminates, rewardlist, rewardlist + gamma * q_prime)
                q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                with tf.GradientTape() as tape:
                    q = tf.math.reduce_sum(
                        policy(np.atleast_2d(statelist).astype('float32')) * tf.one_hot(actionlist, 441),
                        axis=1)
                    loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                # Update the policy network weights using ADAM
                variables = policy.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                loss_history.append(loss.numpy())
            else:
                # don't do anything if we don't have enough memories to construct a batch
                loss_history.append(0)

            # to prevent the proverbial 'dog chasing its own tail' of updating the policy network every time, we only update
            # the target network every update_freq times
            if episode_step_count % update_freq == 0:
                policy_params = policy.trainable_variables
                optimal_params = optimal_policy.trainable_variables

                for pvar, ovar in zip(policy_params, optimal_params):
                    ovar.assign(pvar.numpy())

            # new episode if a terminal state is reached
            if terminated:
                break

        reward_history[episode] = episode_return
        # calculate average reward of previous 100 episodes
        average_reward = reward_history[max(0, episode - 100):(episode+1)].mean()

        if episode % log_freq == 0:
            print("Episode: " + str(episode) + " Episode Reward: " + str(episode_return) + " Average Reward: " + str(average_reward))

    if save_models:
        save_model(policy, model_name)

        average_rewards = [reward_history[max(0, episode - 100):(episode+1)].mean() for episode in range(episodes)]

        plt.plot(reward_history)
        plt.title(model_name)
        plt.savefig(model_name + datetime.now().strftime(" %d %m %H:%M:%S") + ".png")

        plt.plot(average_rewards)
        plt.title(model_name + " average over time")
        plt.savefig(model_name + " average " + datetime.now().strftime("%d %m %H:%M:%S") + ".png")
        plt.close()

    test_env.close()

def validate(model_path):
    model_to_validate = load_model(model_path)
    write_to_file = True
    eval_episodes = 3

    eval_env = TorcsEnv(vision=vision, throttle=True, gear_change=False)
    returns = []

    for episode in range(eval_episodes):
        curr_state = eval_env.reset(relaunch=True)
        episode_return = 0

        for timestep in range(500):
            # collect best action
            action = np.argmax(model_to_validate(np.atleast_2d(np.atleast_2d(curr_state).astype('float32'))))

            # apply action
            next_state, reward, terminated = eval_env.step(action)

            # note reward
            episode_return += reward

            if terminated or timestep >= 499:
                returns.append(episode_return)
                break

            # s <- s'
            curr_state = next_state

    eval_env.close()

    if write_to_file:
        f = open(model_path + ".log", "w")
        f.write("Average: " + str(np.sum(returns)/len(returns)) + "\n")

        for episode in range(len(returns)):
            f.write("Validation Episode: " + str(episode) + " Episode Return: " + str(returns[episode]) + "\n")

        f.close()

    return np.sum(returns)/len(returns)

# function calls
train('DQN2')
DQN_model = load_model('DQN')
train('DQFE', pre_trained=True, trained_agent=DQN_model)
# validate('DQN2')