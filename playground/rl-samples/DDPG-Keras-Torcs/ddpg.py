from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

import gym
from gym import wrappers

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import pprint as pp

OU = OU()       #Ornstein-Uhlenbeck Process

def runModel(args):    #1 means Train, 0 means simply Ruargsn
    train_indicator = 1
    if args["mode"] is not 'training':
        train_indicator = 0
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    np.random.seed(int(args['random_seed']))
    env = gym.make(args['env'])
    env.seed(int(args['random_seed']))
    tf.set_random_seed(int(args['random_seed']))
    # SET THE SEED ON THE ENVIRONMENT --> i.e. send it to webots for deterministic simulation


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high




    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, action_bound, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    #TODO CHANGE REPLAY BUFFER TO OUR NEW IMPLEMENTATION WITH TERMINAL BASED RECALL
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("{}/actormodel.h5".format(args["network_dir"]))
        critic.model.load_weights("{}/criticmodel.h5".format(args["network_dir"]))
        actor.target_model.load_weights("{}/actormodel.h5".format(args["network_dir"]))
        critic.target_model.load_weights("{}/criticmodel.h5".format(args["network_dir"]))
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("REGIS Experiment Start.")
    for i in range(episode_count):
        try:
            s_t = env.reset()
        except:
            import ipdb; ipdb.set_trace()
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        #if np.mod(i, 3) == 0:
        #    ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        #else:

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE

            a_t = actor.model.predict(np.reshape(s_t, (1, state_dim)))
            noise_t = train_indicator * max(epsilon, 0) * OU.function(a_t[0],  0.0 , 0.060, 0.030)

            a_t = a_t + noise_t

            # WE USES THREE DIFFERENT NOIS FOR THE DIFFERENT ACTION DIMENSION WE WANT TO USE ONLY ONE
            #noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            #noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)
            #TODO NOTE: CHANGED, check if OU.function works in multidimensional case
            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
            #a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            #a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            #a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            s_t1, r_t, done, info = env.step(a_t[0])
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add to replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if train_indicator:
                print("Now we save model")
                actor.model.save_weights("{}/actormodel.h5".format(args["network_dir"]), overwrite=True)
                with open("{}/actormodel.json".format(args["network_dir"]), "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("{}/criticmodel.h5".format(args["network_dir"]), overwrite=True)
                with open("{}/criticmodel.json".format(args["network_dir"]), "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--mode', help='choose the mode can be training or testing', default='training')
    parser.add_argument('--network-dir', help='directory for storing network', default='./results/graph')
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    args = vars(parser.parse_args())

    pp.pprint(args)
    runModel(args)
