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
from gym.envs.registration import register


from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import pprint as pp

OU = OU()       #Ornstein-Uhlenbeck Process

def getEnv(args):
    env = None
    try:
        env = gym.make(args['env'])
    except:
        envName = args['env']
        # Algorithmic
        # ----------------------------------------
        register(
            id=envName,
            entry_point='gym.envs.webots:Regis',
            kwargs=dict(
                action_dim = int(args["action_dim"]),
                repeat = int(args["action_repeat"])
            )
        )
        env = gym.make(envName)
    return env

def runModel(args):
    MAX_EPISODES = int(args["max_episodes"])
    MAX_STEPS_PER_EPISODE = int(args["max_episode_len"])
    SAVE_DIR = args["network_dir"]
    IS_TRAINING = 1
    if args["mode"] is not 'training':
        IS_TRAINING = 0

    #BUFFER_SIZE = int(args["buffer_size"])
    BUFFER_TYPE = args["buffer_type"]
    BATCH_SIZE = int(args["minibatch_size"])
    BUFFER_SIZE = BATCH_SIZE*8

    LRA = args["actor_lr"] #Learning rate for Actor
    LRC = args["critic_lr"] #Lerning rate for Critic
    EXPLORE_DECREASE_RATE = args["exploration_dr"]
    GAMMA = args["gamma"]
    TAU = args["tau"] #Target Network HyperParameters

    # SET THE SEED ON THE ENVIRONMENT --> i.e. send it to webots for deterministic simulation
    RANDOM_SEED = int(args['random_seed'])
    env = getEnv(args)
    env.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)




    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

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
    buff = ReplayBuffer(BUFFER_SIZE,RANDOM_SEED)    #Create replay buffer

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("{}/actormodel.h5".format(SAVE_DIR))
        critic.model.load_weights("{}/criticmodel.h5".format(SAVE_DIR))
        actor.target_model.load_weights("{}/actormodel.h5".format(SAVE_DIR))
        critic.target_model.load_weights("{}/criticmodel.h5".format(SAVE_DIR))
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("REGIS Experiment Start.")
    for i in range(MAX_EPISODES):
        try:
            s_t = env.reset()
        except:
            import ipdb; ipdb.set_trace()

        #print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.size()))

        #if np.mod(i, 3) == 0:
        #    ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        #else:

        animate_this_episode = ((i % int(args["render_every"]) == 0) and args["render_env"])
        total_reward = 0.
        if animate_this_episode:
            env.render(mode="human")
        else:
            env.render(mode="bot")
        for j in range(MAX_STEPS_PER_EPISODE):

            loss = 0
            epsilon -= EXPLORE_DECREASE_RATE


            a_t = actor.model.predict(np.reshape(s_t, (1, state_dim)))
            noise_t = max(epsilon, 0) * OU.function(a_t[0],  0.0 , 0.060, 0.030)
            a_t = a_t + noise_t
            # print(a_t)

            s_t1, r_t, done, info = env.step(a_t[0])
            #if np.mod(step, BATCH_SIZE*8) < BATCH_SIZE*4 :
            #    buff.add(s_t, a_t[0], -r_t, s_t1, done)      #Add to replay buffer
            #else:
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add to replay buffer

            #Do the batch update
            batch = None
            #print "buffer type {}". format(BUFFER_TYPE)
            if BUFFER_TYPE != "terminal":
                batch = buff.getBatch(BATCH_SIZE)
            else:
                batch = buff.getBatchTerminals(BATCH_SIZE)
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

            if (IS_TRAINING):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            #if np.mod(j,100) == 0:
            #    print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 1) == 0: # Save weights every episodes
            if IS_TRAINING:
                print("Now we save model")
                actor.model.save_weights("{}/actormodel.h5".format(SAVE_DIR), overwrite=True)
                with open("{}/actormodel.json".format(SAVE_DIR), "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("{}/criticmodel.h5".format(SAVE_DIR), overwrite=True)
                with open("{}/criticmodel.json".format(SAVE_DIR), "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=1e-6) # 1e-4
    parser.add_argument('--critic-lr', help='critic network learning rate', default=1e-4) # 1e-3
    parser.add_argument('--exploration_dr', help='exploration decrease rate', default=1e-5)

    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99) # 0.99
    parser.add_argument('--tau', help='soft target update parameter', default=1e-3) # 1e-3
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1e5) # 1e6 : old
    parser.add_argument('--buffer-type', help='type of buffer can be normal or terminal', default="normal")
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64) # 64 : old

    # Env parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--action_dim', help='Action dimensions', default=11)
    parser.add_argument('--action_repeat', help='Action repeat', default=4)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--render-every', help='render the environment every N episodes', default=10)

    # run parameters

    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1e5)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--mode', help='choose the mode can be training or testing', default='training')
    parser.add_argument('--network-dir', help='directory for storing network', default='./results/graph')

    args = vars(parser.parse_args())

    pp.pprint(args)
    runModel(args)
