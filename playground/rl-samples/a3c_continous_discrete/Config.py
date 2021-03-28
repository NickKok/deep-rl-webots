# -*- coding: utf-8 -*-
import gym
import multiprocessing
from gym.envs.registration import register
import numpy as np

def normalizeInput(config,stabilize_obs = 200, expected_obs = 200):
    env = config.env
    state_dim = config.N_S
    action_dim = config.N_A
    try:
        _ = env.reset()
    except:
        import ipdb; ipdb.set_trace()
    # We want to gather some observation here to get better output than expected.
    # We give no actions. So zero. and get the outputs until its done

    state_mu = np.zeros(state_dim)
    state_sigma = 1+np.zeros(state_dim)
    S_ = 0*np.ndarray([expected_obs,state_dim])
    i = 0
    print("Gathering some observation to normalize inputs")
    while i < stabilize_obs:
        s_, r, done, info = env.step(np.zeros(action_dim))
        i = i+1
    i=0
    while i < expected_obs:
        s_, r, done, info = env.step(np.zeros(action_dim))
        S_[i,:] = s_
        i = i+1
    # Use symmetry :
    #TODO CHECK
    S = None
    # # We copy the array
    # leftS = np.copy(S_)
    # rightS = np.copy(S_)
    # # We erase half of it
    # leftRange = range(np.shape(S_)[1]/2,np.shape(S_)[1])
    # rightRange = range(np.shape(S_)[1]/2)
    # leftS[:,rightRange]=0
    # rightS[:,leftRange]=0
    # # We sum both half
    # leftS[:,leftRange] += rightS[:,rightRange]
    # rightS[:,rightRange] += leftS[:,leftRange]
    # # And create back the array
    # S = leftS+rightS
    # S /= 2

    state_mu = np.mean(S_)
    state_sigma = np.std(S_)
    return state_mu, state_sigma
    print("Done! lets start learning baby !")

def getEnv(NAME,PORT,ACTION_DIM,ACTION_REPEAT,RANDOM_SEED):
    print("Creating env {}, action_dim={}, action_repeat={}, port={}".format(NAME,ACTION_DIM,ACTION_REPEAT,RANDOM_SEED))
    env = None
    #try:
        #env = gym.make(NAME)
    #except:
    envName = "{}{}".format(NAME,PORT)
    # Algorithmic
    # ----------------------------------------
    register(
        id=envName,
        entry_point='gym.envs.webots:Regis',
        kwargs=dict(
            action_dim = int(ACTION_DIM),
            repeat = int(ACTION_REPEAT),
            port = PORT # TODO UPDATE THE PORT FROM THE REGIS UPDATED IN THE DESKTOP COMPUTER
        )
    )
    env = gym.make(envName)
    env.seed(RANDOM_SEED)
    return env

class Config:
    mode = 'continuous'
    # mode = 'discrete'
    MAX_EP_STEP = 400 #TODO Unused
    GLOBAL_EP = 0
    GLOBAL_NET_SCOPE = 'Global_Net'

    NORMALIZE_MEAN = 1.0
    NORMALIZE_STD = 1.0
    N_S = 1
    N_A = 1
    ACTION_BOUND = [-1,1]
    ACTION_GAP = 2


    env = None
    def __init__(self,game="Regis-v",args=None):
        if args is None:
            raise ValueError('No arguments passed to global config')
        print('Game: {}'.format(game))
        self.GAME = game
        self.INIT_PORT = args["communication_port"]
        self.LR_A = float(args["actor_lr"])
        self.LR_C = float(args["critic_lr"])
        self.ENTROPY_BETA = float(args["entropy_beta"])
        self.GAMMA = float(args["gamma"])
        self.REWARD_SCALING = float(args["reward_scaling"])
        self.MODEL_DIR = args["save_dir"]
        self.MAX_GLOBAL_EP = int(args["max_episodes"])
        self.UPDATE_GLOBAL_ITER = int(args["update_batch_iter"])
        self.ACTOR_NETWORK_TYPE = int(args["actor_network_type"])
        self.CRITIC_NETWORK_TYPE = int(args["critic_network_type"])
        self.INPUT_SPACE_TYPE = int(args["input_space_type"])
        self.TEMPORAL_WINDOW = int(args["temporal_window"])

        self.N_WORKERS = int(args["n_workers"])
        self.ACTION_REPEAT = int(args["action_repeat"])
        self.RANDOM_SEED = int(args["random_seed"])
        self.FORGET_WINDOW_SIZE = int(args["forget_window_size"])

        self.env = getEnv(self.GAME,self.INIT_PORT,args["action_dim"],self.ACTION_REPEAT,self.RANDOM_SEED)
        self.N_S = self.env.observation_space.shape[0]
        if self.mode == 'discrete':  # Note：The action_space of CartPole-v0 does not contain attribute 'shape'
            self.N_A = self.env.action_space.n
        elif self.mode == 'continuous':  # Note: The action of Pendulum-v0 is a list with shape (1,)
            self.N_A = self.env.action_space.shape[0]
            assert(args["action_dim"] == self.N_A, "We have a problem, you want action_dim={}!={}".format(args["action_dim"], self.N_A))
            self.ACTION_BOUND = [self.env.action_space.low, self.env.action_space.high]
            self.ACTION_GAP = self.env.action_space.high - self.env.action_space.low
