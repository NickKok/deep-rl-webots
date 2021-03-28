# -*- coding: utf-8 -*-
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib
import argparse
import pprint as pp

from Config import Config
from Config import getEnv, normalizeInput
from myUtils import lazy_property, dense, variable_summaries, variable_summaries_history, variable_summaries_layer, variable_summaries_scalar

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

GLOBAL_RUNNING_R = []  # record the history scores

from Network import ACNet
from Worker import Worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='provide arguments for your RL(A3C) agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001) # 1e-4
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001) # 1e-3
    parser.add_argument('--entropy-beta', help='exploration rate', default=0.00001)
    parser.add_argument('--update-batch-iter', help='number of iteration before update of global network', default=400)
    parser.add_argument('--n-workers', help='number of parallel workers (not more then n cpu)', default=1)
    parser.add_argument('--actor-network-type', help='type of neural network for actor, check get_mu_sigma method', default=1)
    parser.add_argument('--critic-network-type', help='type of neural network for critic, check get_mu_sigma method', default=1)
    parser.add_argument('--input-space-type', help='type of input space 1 is "position", 2 is "velocity"', default=2)
    parser.add_argument('--temporal-window', help='number of consecutive frame to use for learning', default=1)
    parser.add_argument('--reward-scaling', help='reward scaling', default=1)
    parser.add_argument('--forget-window-size', help='reward scaling', default=0)



    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.6) # 0.99
    #parser.add_argument('--tau', help='soft target update parameter', default=1e-3) # 1e-3
    #parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1e5) # 1e6 : old
    #parser.add_argument('--buffer-type', help='type of buffer can be normal or terminal', default="normal")
    #parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64) # 64 : old

    # Env parameters
    #parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--action-dim', help='Action dimensions', default=23)
    parser.add_argument('--action-repeat', help='Action repeat', default=10)
    parser.add_argument('--communication-port', help='Initial port used for communication', default=5662)
    #parser.add_argument('--render-env', help='render the gym env', action='store_true')
    #parser.add_argument('--render-every', help='render the environment every N episodes', default=10)

    # run parameters
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    #parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1e5)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    #parser.add_argument('--mode', help='choose the mode can be training or testing', default='training')
    parser.add_argument('--save-dir', help='directory for storing network', default='./log')

    args = vars(parser.parse_args())

    pp.pprint(args)

    # -----------------
    # Global parameters
    # -----------------
    tf.set_random_seed(int(args["random_seed"]))
    np.random.seed(int(args["random_seed"]))

    CHECKPOINT_DIR = os.path.join(args["save_dir"], "checkpoints")
    SUMMARY_DIR = os.path.join(args["save_dir"], "summary")
    if not os.path.exists(CHECKPOINT_DIR):
      os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(SUMMARY_DIR):
      os.makedirs(SUMMARY_DIR)

    # ----------------------------
    # Check if previous TF session
    # ----------------------------
    PREVIOUS_SESSION = tf.train.latest_checkpoint(CHECKPOINT_DIR)

    # ------------------------------------
    # Create config and env initialization
    # ------------------------------------
    # Config creation
    # If PREVIOUS_SESSION existing the env created is on port+1
    CONFIG = None
    if PREVIOUS_SESSION:
        args["communication_port"] += 1
    CONFIG = Config("Regis-v",args=args)


    # Normalize state input

    state_mu = 0.0
    state_sigma = 1.0

    if CONFIG.INPUT_SPACE_TYPE == 2:
        state_mu = 0.1802480765585894
        state_sigma = 0.36703051177573315
    if CONFIG.INPUT_SPACE_TYPE == 1:
        state_mu=-0.128282137
        state_sigma=0.668751003423
    if CONFIG.INPUT_SPACE_TYPE == 0:
        state_mu=0
        state_sigma=1.0




    # if not PREVIOUS_SESSION:
    #     print "No previous session found, normalizing inputs"
    #     state_mu, state_sigma = normalizeInput(CONFIG,200,200)
    #     print "Ok normalization finished : mu={}, sigma={}".format(state_mu,state_sigma)

    CONFIG.NORMALIZE_MEAN = state_mu
    CONFIG.NORMALIZE_STD = state_sigma

    # --------------
    # Create networks
    # --------------
    SESS = tf.Session()
    COORD = tf.train.Coordinator()

    unique_mutex = threading.Lock()
    with tf.device("/cpu:0"):
        global_ac_name = CONFIG.GLOBAL_NET_SCOPE
        GLOBAL_AC = ACNet(global_ac_name, CONFIG)  # we only need its params

    summary_writer = tf.summary.FileWriter(SUMMARY_DIR,SESS.graph)

    workers = []
    workernames = []
    networks = []
    for i in range(CONFIG.N_WORKERS):
        name = 'W_%i' % i  # worker name
        workernames.append(name)
        networks.append(ACNet(name, CONFIG, GLOBAL_AC, summary_writer))

    # --------------------
    # Graph initialization
    # --------------------
    # All Tensorflow graphs have been created. We can safely initialize now
    SESS.run(tf.global_variables_initializer())
    # After initialization we create the summaries and saver
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=10)
    # Now load previous session if present
    if PREVIOUS_SESSION:
        print("Found existing model, loading checkpoint: {}".format(PREVIOUS_SESSION))
        CONFIG.GLOBAL_EP = int(PREVIOUS_SESSION.split('-')[-1])
        print("Start at episode {}".format(CONFIG.GLOBAL_EP))
        saver.restore(SESS, PREVIOUS_SESSION)


    # Finally we create the workers. This ends the initilaization procedure

    if PREVIOUS_SESSION:
        for i,(name,net) in enumerate(zip(workernames,networks)):
            env = CONFIG.env
            saver_ = saver
            if i is not 0:
                saver_ = None
                env = getEnv(CONFIG.GAME,CONFIG.INIT_PORT+i,CONFIG.N_A,CONFIG.ACTION_REPEAT,CONFIG.RANDOM_SEED)
            else:
                print("first worker, we reuse existing env used for init")
            workers.append(Worker(name, net, SESS, COORD, CONFIG, unique_mutex, summary_writer, env=env, saver=saver_))
    else:
        for i,(name,net) in enumerate(zip(workernames,networks)):
            env = CONFIG.env
            saver_ = saver
            if i is not 0:
                saver_ = None
            	env = getEnv(CONFIG.GAME,CONFIG.INIT_PORT+i+1,CONFIG.N_A,CONFIG.ACTION_REPEAT,CONFIG.RANDOM_SEED)
            workers.append(Worker(name, net, SESS, COORD, CONFIG, unique_mutex, summary_writer, saver=saver_, env=env))

    # ---------------
    # Actual learning
    # ---------------
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda: worker.work())
        t.start()
        worker_threads.append(t)

    COORD.join(worker_threads)
    #plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    #plt.xlabel('step')
    #plt.ylabel('Total moving reward')
    #plt.show()
