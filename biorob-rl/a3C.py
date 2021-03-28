# -*- coding: utf-8 -*-
import threading
import tensorflow as tf
import numpy as np
import gym

import re
import os
import sys
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
    parser.add_argument('--n-workers', help='number of parallel workers (not more then n cpu)', default=1)
    parser.add_argument('--actor-network-type', help='type of neural network for actor, check get_mu_sigma method', default=1)
    parser.add_argument('--critic-network-type', help='type of neural network for critic, check get_mu_sigma method', default=1)
    parser.add_argument('--input-space-type', help='type of input space 1 is "position", 2 is "velocity"', default=0)
    parser.add_argument('--temporal-window', help='number of consecutive frame to use for learning', default=1)
    parser.add_argument('--reward-scaling', help='reward scaling', default=1)
    parser.add_argument('--forget-window-size', help='forget window size', default=0)

    parser.add_argument('--world', dest='world', action='store', default='')


    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.6) # 0.99
    # Env parameters
    parser.add_argument('--action-dim', help='Action dimensions', default=23)
    parser.add_argument('--action-repeat', help='Action repeat', default=1)
    parser.add_argument('--communication-port', help='Initial port used for communication', default=5662)
    # run parameters
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--mode', help='choose the mode can be training or testing', default='training')
    parser.add_argument('--save-dir', help='directory for storing network', default='./log')
    parser.add_argument('--gpu', help='whether to use GPU', default=False)

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

    workers = []
    workernames = []
    networks = []


    device = 'cpu:0'

    if(args['gpu']):
        device = 'gpu:1'

    unique_mutex = threading.Lock()
    with tf.device("/{}".format(device)):
        global_ac_name = CONFIG.GLOBAL_NET_SCOPE
        GLOBAL_AC = ACNet(global_ac_name, CONFIG)  # we only need its params

    summary_writer = tf.summary.FileWriter(SUMMARY_DIR,SESS.graph)

    for i in range(CONFIG.N_WORKERS):
        name = 'W_%i' % i  # worker name
        if(i == 0):
            workernames.append(name)
            networks.append(ACNet(name, CONFIG, GLOBAL_AC, summary_writer))
        else:
            with tf.device("/{}".format(device)):
                workernames.append(name)
                networks.append(ACNet(name, CONFIG, GLOBAL_AC, summary_writer))

    # --------------------
    # Graph initialization
    # --------------------
    # All Tensorflow graphs have been created. We can safely initialize now
    SESS.run(tf.global_variables_initializer())
    # After initialization we create the summaries and saver

    variables_to_restore = [var for var in tf.global_variables()
                        if var.name.startswith('W_0')]

    saver = tf.train.Saver(variables_to_restore, keep_checkpoint_every_n_hours=2, max_to_keep=10)
    # Now load previous session if present
    if PREVIOUS_SESSION:
        print("Found existing model, loading checkpoint: {}".format(PREVIOUS_SESSION))
        CONFIG.GLOBAL_EP = int(PREVIOUS_SESSION.split('-')[-1])
        print("Start at episode {}".format(CONFIG.GLOBAL_EP))
        saver.restore(SESS, PREVIOUS_SESSION)


    # Finally we create the workers. This ends the initilaization procedure
    # Check if testing
    
    if (CONFIG.MODE == "testing" or CONFIG.MODE == "openloop") and not PREVIOUS_SESSION:
        print("You want to test but we don't find any previous session, maybe train first")
        sys.exit(1)
    if (CONFIG.MODE == "testing" or CONFIG.MODE == "openloop") and CONFIG.N_WORKERS != 1:
        print("You want to test with more than one workers, try with --n-worker=1")
        sys.exit(1)

    if PREVIOUS_SESSION:
        for i,(name,net) in enumerate(zip(workernames,networks)):
            env = CONFIG.env
            saver_ = saver
            if i is not 0:
                saver_ = None
                port = CONFIG.INIT_PORT+i
                world = re.sub(r'\d\d\d\d',r"{}".format(port), CONFIG.WORLD)
                env = getEnv(CONFIG.GAME,port,CONFIG.N_A,CONFIG.ACTION_REPEAT,CONFIG.RANDOM_SEED,world)
            else:
                print("first worker, we reuse existing env used for init")
            workers.append(Worker(name, net, SESS, COORD, CONFIG, unique_mutex, summary_writer, env=env, saver=saver_))
    else:
        for i,(name,net) in enumerate(zip(workernames,networks)):
            env = CONFIG.env
            saver_ = saver
            if i is not 0:
                port = CONFIG.INIT_PORT+i+1
                saver_ = None
                world = re.sub(r'\d\d\d\d',r"{}".format(port), CONFIG.WORLD)
                env = getEnv(CONFIG.GAME,port,CONFIG.N_A,CONFIG.ACTION_REPEAT,CONFIG.RANDOM_SEED,world)
            workers.append(Worker(name, net, SESS, COORD, CONFIG, unique_mutex, summary_writer, saver=saver_, env=env))


    # ---------------
    # Actual learning/training
    # ---------------
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda w: w.work(), args = (worker,))
        t.start()
        worker_threads.append(t)

    COORD.join(worker_threads)
    #plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    #plt.xlabel('step')
    #plt.ylabel('Total moving reward')
    #plt.show()
