#/home/efx/.local/opt/anaconda2/envs/opensim-rl/bin/python
# Derived from keras-rl
import opensim as osim

import argparse


# Command line parameters
WORLD = '/home/efx/Development/PHD/AiriNew/humanWebotsNmm/controller_current/webots/worlds/3D_selfCollision_running.wbt'
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=300000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--world', dest='world', action='store', default=WORLD)
#TODO : Add parameters related to environment creation, action_dim, port, ...
#TODO : Add parameters related to max number of step, number of instances, ...

print(osim.__path__)

args = parser.parse_args()



import numpy as np
import sys
import pickle

import keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

from Config import Config
from Config import getEnv, normalizeInput


from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


#sys.path = ["../"]+sys.path
from osim.env import *

import osim.env.arm as arm

from keras.optimizers import RMSprop

import math

import matplotlib.pylab as plt




class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.reward = []

    def on_batch_end(self, batch, logs={}):

        self.losses.append(logs.get('observation'))
        self.reward.append(logs.get('reward'))
        
        
        
# Load walking environment
#env =arm.ArmEnv(args.visualize)

#env = getEnv('Regis-v',5662,8,20,0)
#WORLD = '/home/efx/Development/PHD/AiriNew/humanWebotsNmm/controller_current/webots/worlds/3D_RL_slope.wbt'
#WORLD = '/home/efx/Development/PHD/AiriNew/humanWebotsNmm/controller_current/webots/worlds/3D_RL_hard.wbt'
#WORLD = '/home/efx/Development/PHD/AiriNew/humanWebotsNmm/controller_current/webots/worlds/3D_RL.wbt'


env = getEnv(
        name='Regis-v',
        port=5662,
        action_dim=18,
        action_repeat=20,
        random_seed=0,
        world=args.world)


nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

# Next, we build a very simple model.
# Create networks for DDPG

from keras import initializers

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(200))
actor.add(Activation('relu'))
actor.add(Dense(200))
actor.add(Activation('relu'))
actor.add(Dense(200))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
#actor.add(Activation('relu'))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(200)(x)
x = Activation('relu')(x)
x = Dense(200)(x)
x = Activation('relu')(x)
x = Dense(200)(x)
x = Activation('relu')(x)
x = Dense(100)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

history = LossHistory()

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)

random_process = OrnsteinUhlenbeckProcess(theta=.09, mu=0., sigma=.2, size=env.action_dim, sigma_min=0.05, n_steps_annealing=1e4)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.96, target_model_update=1e-2,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    # agent.load_weights(args.model)

    agent.fit(env, nb_steps=nallsteps, visualize=args.visualize, verbose=1, nb_max_episode_steps=5000, log_interval=100, callbacks=[history])
    
    lst6 = [item for item in history.reward]
    with open('figure2_test.data', 'wb') as filehandle: 
        pickle.dump(lst6, filehandle)
        
    plt.plot(lst6)
    plt.show()
    #print history.losses
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)
    



if not args.train:

    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=True, callbacks=[history])
    
    lst6 = [item for item in history.reward]
    
    with open('figure2_train.data', 'wb') as filehandle: 
        pickle.dump(lst6, filehandle)

    #for i in range(5000):
    #    observation, reward, done, info = env.step(env.action_space)#env.action_space.sample())
    #    print(observation)
    #agent.forward
