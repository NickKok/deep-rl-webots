# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#import cPickle as pickle # For python 2.7, see https://askubuntu.com/questions/742782/how-to-install-cpickle-on-python-3-4
import pickle
import copy
import subprocess


from myUtils import lazy_property, dense, variable_summaries_history, variable_summaries_layer, variable_summaries_scalar, getActionList, getNonZeroActionList,episodeBuffer


class ACNet(object):
    def __init__(self, scope, config, globalAC=None, summary_writer=None):
        self.config = config
        self.globalAC = globalAC
        self.action_dim = self.config.N_A
        self.state_dim = self.config.N_S
        self.summary_writer = summary_writer
        self.is_local_net = globalAC is not None
        self.name = scope
        if scope == self.config.GLOBAL_NET_SCOPE:  # get global network
            self.trial_buffers_max_size = 30
            self.best_trial_buffer = []
            self.best_trial_buffers = []
            self.best_trial_fitness = 0.0
            self.best_trial_fitnesses = []
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(shape=[None, self.state_dim, self.config.TEMPORAL_WINDOW], dtype=tf.float32, name="S")
                # initialize actor-net according to different config.mode
                self.actions = self.action_get_current(reuse=True)
                self.value_get_current
                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(shape=[None, self.state_dim, self.config.TEMPORAL_WINDOW], dtype=tf.float32)
                self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
                self.action_input_target = tf.placeholder(tf.float32, [None, self.action_dim])
                self.value_target = tf.placeholder(tf.float32, [None, 1])
                self.advantages = tf.placeholder(tf.float32, [None, 1])

                self.actions = self.action_get_current(reuse=False)
                self.value_get_current

                self.TD_loss
                self.TL_loss
                self.critic_loss
                self.actor_loss

                #Get gradients from local network using local losses
                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

                self.optimizer_actor = tf.train.AdamOptimizer(self.config.LR_A,epsilon=1e-01)
                self.optimizer_critic = tf.train.AdagradOptimizer(self.config.LR_C)

                #Interact with global network
                self.pull_params
                self.pull_critic_params
                self.pull_actor_params
                self.push_params
                self.push_critic_params
                self.push_actor_params

                with tf.variable_scope('rewards'):
                    if self.name == "W_0":
                        tf.summary.scalar("mean", tf.reduce_mean(self.value_target))
                        tf.summary.histogram("targets", self.value_target)
                    # variable_summaries_scalar(self.value_target,self.is_local_net)

                if self.summary_writer is not None:
                    if self.name == "W_0":
                        self.summaries = tf.summary.merge_all()

    def loadBestInvid(self):
        bashCommand = "ls -l {}/".format(self.config.SAVE_DIR) + " | grep 'bestOpen' | awk '{print $9}' | sed -e 's/_/ /g' | awk '{print $2}' | sed -e 's/\./ /g' | awk '{print $1}'"
        #print bashCommand
        stdout, _ = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE).communicate()
        np.array([int(i) for i in stdout.split('\n')[:-1]])
	#print '{}/bestOpenLoopActor_{}.pickle'.format(self.config.SAVE_DIR,max(np.array([int(i) for i in stdout.split('\n')[:-1]])))
        with open('{}/bestOpenLoopActor_{}.pickle'.format(self.config.SAVE_DIR,max(np.array([int(i) for i in stdout.split('\n')[:-1]]) )),'rb') as input_file: 
            self.best_trial_buffer = pickle.load(input_file)
            self.best_trial_fitness = self.best_trial_buffer[-1][2] # Last reward is the fitness
            self.best_trial_buffers.append(self.best_trial_buffer)
            self.best_trial_fitnesses.append(self.best_trial_fitness)
        

    def updateInvid(self,episode_buffer, actual_fitness):
        # We want to save to file the best (open loop) invid so that it can be replayed later
        # This can be used to offline test the convergence of the actor. 
        best_fitness = -10000.0
        if(self.best_trial_fitness != 0):
            best_fitness = self.best_trial_fitness

        # We update the buffers of less good indivd
        if(self.best_trial_fitness != actual_fitness):
            self.best_trial_buffers.append(copy.deepcopy(episode_buffer))
            self.best_trial_fitnesses.append(actual_fitness)

        if(actual_fitness >= best_fitness):
            # for i in episode_buffer:
            #     print("{}".format(i[episodeBuffer.a]))
            self.best_trial_buffer = copy.deepcopy(episode_buffer)
            self.best_trial_fitness = actual_fitness
            with open(r"{}/bestOpenLoopActor_{}.pickle".format(self.config.SAVE_DIR,self.config.GLOBAL_EP), "wb") as output_file:
                pickle.dump(self.best_trial_buffer, output_file)

        # If fitness is big we fluzh the best directino buffer.
        # if(actual_fitness > 1.2*best_fitness):
        #     self.best_trial_fitnesses = []
        #     self.best_trial_buffers = []
        #     self.best_trial_fitnesses.append(copy.deepcopy(actual_fitness))
        #     self.best_trial_buffers.append(episode_buffer)
        #     return 




        superSort =lambda X,Y: [x for _ , x in sorted(zip(Y,X), key=lambda pair: pair[0])]

        self.best_trial_fitnesses = superSort(self.best_trial_fitnesses,self.best_trial_fitnesses)
        self.best_trial_buffers = superSort(self.best_trial_buffers,self.best_trial_fitnesses)

        #self.best_trial_fitnesses.reverse()
        #self.best_trial_buffers.reverse()
        if(len(self.best_trial_fitnesses) > self.trial_buffers_max_size):
            self.best_trial_fitnesses.pop()
            self.best_trial_buffers.pop()

    def getBestDirection(self):
        if(len(self.best_trial_fitnesses) < 2):
            return []
        best = getNonZeroActionList(self.best_trial_buffer)
        Fbest = self.best_trial_fitness


        bads = [getNonZeroActionList(buf) for buf in self.best_trial_buffers]
        Fbads = [f for f in self.best_trial_fitnesses]

        #tot = sum([(best - bad) * (Fbest - Fbad)/Fbest for Fbad, bad in zip(Fbads, bads)])[0]
        #tot = sum([(best - bad) * (Fbest - Fbad) for Fbad, bad in zip(Fbads, bads)])[0]
        #tot = sum([(best - bad) for Fbad, bad in zip(Fbads, bads)])[0]
        #scale = len(Fbads)
        tot   = sum([(best - bad) * (Fbest - Fbad) for Fbad, bad in zip(Fbads, bads)])[0]
        scale = sum([Fbest - Fbad for Fbad in Fbads])*len(Fbads)
        return tot/scale



    @lazy_property
    def actor_loss(self):
        with tf.variable_scope('actor_loss'):
            actor_loss = tf.reduce_mean(tf.square(self.TL_loss))#*tf.reduce_sum(0.1*tf.reduce_mean(self.advantages,axis=0))
            if self.name == "W_0":
                variable_summaries_scalar(actor_loss,self.is_local_net)
        return actor_loss

    @lazy_property
    def critic_loss(self):
        with tf.variable_scope('critic_loss'):
            critic_loss = tf.reduce_mean(tf.square(self.TD_loss))
            if self.name == "W_0":
                variable_summaries_scalar(critic_loss,self.is_local_net)
            return critic_loss

    @lazy_property
    def TL_loss(self):
        return tf.subtract(self.actions, self.action_input_target)

    @lazy_property
    def TD_loss(self):
        return tf.subtract(self.value_target, self.value_get_current)

    @lazy_property
    def value_get_current(self):
        with tf.variable_scope('critic'):
            w_i = tf.random_uniform_initializer(0., 0.1)
            type = self.config.CRITIC_NETWORK_TYPE
            if type == 1:
                with tf.variable_scope('dense1'):
                    dense1 = dense(self.state_input, 256, [256], w_i, activation=tf.nn.relu)
                    if self.name == "W_0":
                        variable_summaries_layer(dense1,self.is_local_net)
                with tf.variable_scope('dense2'):
                    dense2 = dense(dense1, 1, [1], w_i, activation=None)
                    if self.name == "W_0":
                        variable_summaries_layer(dense2,self.is_local_net)
                return dense2
            elif type == 2:
                with tf.variable_scope('dense1'):
                    dense1 = dense(self.state_input, 512, [512], w_i, activation=tf.nn.relu)
                    variable_summaries_layer(dense1,self.is_local_net)
                with tf.variable_scope('dense2'):
                    dense2 = dense(dense1, 256, [256], w_i, activation=tf.nn.relu)
                    variable_summaries_layer(dense2,self.is_local_net)
                with tf.variable_scope('dense3'):
                    dense3 = dense(dense2, 1, [1], w_i, b_i, activation=None)
                    variable_summaries_layer(dense3,self.is_local_net)
                return dense3
            else:
                with tf.variable_scope('dense1'):
                    dense1 = dense(self.state_input, 128, [128], w_i, activation=tf.nn.selu)
                    variable_summaries_layer(dense1,self.is_local_net)
                with tf.variable_scope('dense2'):
                    dense2 = dense(self.state_input, 128, [128], w_i, activation=tf.nn.selu)
                    variable_summaries_layer(dense2,self.is_local_net)
                with tf.variable_scope('dense3'):
                    dense3 = dense(tf.concat([dense1,dense2],axis=1), 1, [1], w_i, b_i, activation=None)
                    variable_summaries_layer(dense3,self.is_local_net)
                return dense3

    # Note: We need 2 return value here: mu & sigma. So it is not suitable to use lazy_property.
    def action_get_current(self, reuse=False):
        # Graph shared with Value Net
        with tf.variable_scope('actor'):
            w_i = tf.initializers.glorot_normal()
            b_i = tf.zeros_initializer()

            if self.config.ACTOR_NETWORK_TYPE == 1:
                with tf.variable_scope('act_dense1'):
                    dense1 = dense(self.state_input, 512, [512], w_i, activation=tf.nn.relu)
                    if not reuse and self.name == "W_0":
                        variable_summaries_layer(dense1,self.is_local_net)
                with tf.variable_scope('act_dense2'):
                    dense2 = dense(dense1, 256, [256], w_i, activation=tf.nn.relu)
                    if not reuse and self.name == "W_0":
                        variable_summaries_layer(dense2,self.is_local_net)

                if not reuse and self.name == "W_0":
                    tf.contrib.layers.summarize_activation(dense1)
                    tf.contrib.layers.summarize_activation(dense2)
                with tf.variable_scope('mu'):
                    mu = dense(dense2, self.action_dim, None, w_i, activation=None)
                    if not reuse and self.name == "W_0":
                        variable_summaries_history(mu,self.is_local_net)

            elif self.config.ACTOR_NETWORK_TYPE == 11:
                with tf.variable_scope('act_dense1'):
                    dense1 = dense(self.state_input, 512, [512], w_i, activation=tf.nn.relu)
                    if not reuse and self.name == "W_0":
                        variable_summaries_layer(dense1,self.is_local_net)
                with tf.variable_scope('act_dense2'):
                    dense2 = dense(dense1, 256, [256], w_i, activation=tf.nn.relu)
                    if not reuse and self.name == "W_0":
                        variable_summaries_layer(dense2,self.is_local_net)
                with tf.variable_scope('act_dense3'):
                    dense3 = dense(dense2, 256, [256], w_i, b_i, activation=tf.nn.relu)
                    if not reuse and self.name == "W_0":
                        variable_summaries_layer(dense3,self.is_local_net)

                if not reuse and self.name == "W_0":
                    tf.contrib.layers.summarize_activation(dense1)
                    tf.contrib.layers.summarize_activation(dense2)
                    tf.contrib.layers.summarize_activation(dense3)
                with tf.variable_scope('mu'):
                    mu = dense(dense3, self.action_dim, None, w_i, activation=None)
                    if not reuse and self.name == "W_0":
                        variable_summaries_history(mu,self.is_local_net)

            else:
                raise ValueError('Network type "{}" not implemented, should be integer'.format(self.config.ACTOR_NETWORK_TYPE))
            return mu

    @lazy_property
    def pull_actor_params(self):
        pull_actor_params = [tf.assign(l_p, g_p) for g_p, l_p in zip(self.globalAC.actor_params, self.actor_params)]
        return [pull_actor_params]

    @lazy_property
    def pull_critic_params(self):
        pull_critic_params = [tf.assign(l_p, g_p) for g_p, l_p in zip(self.globalAC.critic_params, self.critic_params)]
        return [pull_critic_params]

    @lazy_property
    def pull_params(self):
        pull_actor_params = [tf.assign(l_p, g_p) for g_p, l_p in zip(self.globalAC.actor_params, self.actor_params)]
        pull_critic_params = [tf.assign(l_p, g_p) for g_p, l_p in zip(self.globalAC.critic_params, self.critic_params)]
        return [pull_actor_params, pull_critic_params]

    @lazy_property
    def push_actor_params(self):
        push_actor_params = self.optimizer_actor.apply_gradients(zip(self.actor_grads, self.globalAC.actor_params))
        return [push_actor_params]

    @lazy_property
    def push_critic_params(self):
        push_critic_params = self.optimizer_critic.apply_gradients(zip(self.critic_grads, self.globalAC.critic_params))
        return [push_critic_params]

    @lazy_property
    def push_params(self):
        push_actor_params = self.optimizer_actor.apply_gradients(zip(self.actor_grads, self.globalAC.actor_params))
        push_critic_params = self.optimizer_critic.apply_gradients(zip(self.critic_grads, self.globalAC.critic_params))
        return [push_actor_params, push_critic_params]

    @lazy_property
    def sample_action(self):
        return self.actions
