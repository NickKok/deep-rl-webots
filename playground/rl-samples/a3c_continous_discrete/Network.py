# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from myUtils import lazy_property, dense, variable_summaries_history, variable_summaries_layer, variable_summaries_scalar


class ACNet(object):
    def __init__(self, scope, config, globalAC=None, summary_writer=None):
        self.config = config
        self.globalAC = globalAC
        self.action_dim = self.config.N_A
        self.state_dim = self.config.N_S
        self.summary_writer = summary_writer
        self.is_local_net = globalAC is not None

        if scope == self.config.GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(shape=[None, self.state_dim, self.config.TEMPORAL_WINDOW], dtype=tf.float32, name="S")
                # initialize actor-net according to different config.mode
                if self.config.mode == 'continuous':
                    self.mu, self.sigma = self.get_mu_sigma(reuse=True)
                    self.action_normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
                elif self.config.mode == 'discrete':
                    self.a_prob
                self.value_get_current
                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(shape=[None, self.state_dim, self.config.TEMPORAL_WINDOW], dtype=tf.float32)
                if self.config.mode == 'discrete':
                    self.action_input = tf.placeholder(tf.int32, [None, ])
                elif self.config.mode == 'continuous':
                    self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
                    self.action_input_old = tf.placeholder(tf.float32, [None, self.action_dim])
                self.value_target = tf.placeholder(tf.float32, [None, 1])
                self.advantages = tf.placeholder(tf.float32, [None, 1])


                if self.config.mode == 'continuous':
                    self.mu, self.sigma = self.get_mu_sigma(reuse=False)
                    self.action_normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
                elif self.config.mode == 'discrete':
                    self.a_prob
                self.value_get_current

                self.choose_action

                self.TD_loss
                self.critic_loss
                self.actor_loss

                #Get gradients from local network using local losses
                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)


                #Define optimizer
                #self.optimizer_actor = tf.train.RMSPropOptimizer(self.config.LR_A, name='RMSPropA')
                #self.optimizer_actor = tf.train.AdamOptimizer(self.config.LR_A,epsilon=1e-01)
                self.optimizer_actor = tf.train.AdagradOptimizer(self.config.LR_A)
                #self.optimizer_actor = tf.train.ProximalAdagradOptimizer(self.config.LR_A,l1_regularization_strength=1.0)

                #self.optimizer_critic = tf.train.RMSPropOptimizer(self.config.LR_C, name='RMSPropC')
                #self.optimizer_critic = tf.train.AdamOptimizer(self.config.LR_C,epsilon=1e-01)
                self.optimizer_critic = tf.train.AdagradOptimizer(self.config.LR_C)
                #self.optimizer_critic = tf.train.ProximalAdagradOptimizer(self.config.LR_C,l1_regularization_strength=1.0)


                #Interact with global network
                self.pull_params
                self.pull_critic_params
                self.pull_actor_params
                self.push_params
                self.push_critic_params
                self.push_actor_params
                self.actions

                if self.config.ACTOR_NETWORK_TYPE == 4:
                    self.dense1
                    self.muStance
                    self.sigmaStance
                    self.muSwing
                    self.sigmaSwing

                with tf.variable_scope('rewards'):
                    if self.is_local_net:
                        tf.summary.scalar("mean", tf.reduce_mean(self.value_target))
                        tf.summary.histogram("targets", self.value_target)
                    # variable_summaries_scalar(self.value_target,self.is_local_net)

                if self.summary_writer is not None:
                    with tf.variable_scope('actor'):
                        for i in range(self.action_dim):
                            #family = "Rright Leg"
                            #if i < self.action_dim/2:
                            #    family = "Left Leg"
                            #tf.summary.scalar("action-{}".format(i+1), self.actions[i], family)
                            #with tf.variable_scope('actions'):
                            #    tf.summary.scalar(self.actions[i])
                            #with tf.variable_scope('mu'):
                            #    tf.summary.scalar(self.mu[0,i])
                            #with tf.variable_scope('sigma'):
                            #    tf.summary.scalar(self.sigma[0,i])
                            pass
                    self.summaries = tf.summary.merge_all()

    @lazy_property
    def critic_loss(self):
        with tf.variable_scope('critic_loss'):
            critic_loss = tf.reduce_mean(tf.square(self.TD_loss))
            variable_summaries_scalar(critic_loss,self.is_local_net)
            return critic_loss

    @lazy_property
    def actor_loss(self):
        if self.config.mode == 'discrete':
            log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.action_input, self.action_dim, dtype=tf.float32),
                                     axis=1, keep_dims=True)
            # use entropy to encourage exploration
            exp_v = log_prob * self.TD_loss
            entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
            exp_v = self.config.ENTROPY_BETA * entropy + exp_v
            return tf.reduce_mean(-exp_v)
        elif self.config.mode == 'continuous':
            #dt = 0.01
            #log_prob = self.action_normal_dist.log_prob((self.action_input-self.action_input_old)/dt)
            log_prob = self.action_normal_dist.log_prob(self.action_input)
            #log_prob = (self.action_noise)
            entropy = self.action_normal_dist.entropy()
            beta = self.config.ENTROPY_BETA

            loss = -tf.reduce_mean(log_prob * self.TD_loss + beta*entropy,axis=0)
            #loss = -tf.reduce_mean(log_prob*self.advantages + beta*entropy)

            with tf.variable_scope('actor_loss'):
                variable_summaries_scalar(loss,self.is_local_net)
            return loss

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
                    dense1 = dense(self.state_input, 256, [256], w_i, activation=tf.nn.selu)
                    variable_summaries_layer(dense1,self.is_local_net)
                with tf.variable_scope('dense2'):
                    dense2 = dense(dense1, 1, [1], w_i, activation=None)
                    variable_summaries_layer(dense2,self.is_local_net)
                return dense2
            elif type == 2:
                with tf.variable_scope('dense1'):
                    dense1 = dense(self.state_input, 256, [256], w_i, activation=tf.nn.relu)
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


    def general_actor(self, name, num_inputs, num_outputs, ext = 1, reuse=False): # ext = -1 for flexion net
                                                                     # Extension network : positif if state < state_input
        # This is the first network that will try to implement some notion of :
        # - symmetry. same network used for left and right leg. Only state input changes.
        # - phase based network. two different network groups used for swing and stance.
        # - general pd control.
        # Graph shared with Value Net
        with tf.variable_scope('actor/{}'.format(name)):
            state_dim = num_inputs

            cst_input = tf.constant(np.float32(np.zeros([1,1])))
            #cst_input = tf.constant(np.float32(np.random.randn(1,state_dim)))
            state_input = self.state_input
            state_target = tf.contrib.layers.fully_connected(
                inputs=cst_input,
                num_outputs=state_dim,
                activation_fn=None,
                biases_initializer=tf.random_uniform_initializer(-1.0,1.0),
                scope="stateTarget")

            state = tf.contrib.layers.flatten(state_input)
            actionNet = tf.nn.relu(ext*(state_target - state))

            #assert actionNet.shape == [1,state_dim]

            #w_i = tf.random_normal_initializer(0., 0.1)
            w_i = tf.random_uniform_initializer(0., 0.1)
            with tf.variable_scope('mu'):
                mu = dense(actionNet, num_outputs, None, w_i, None, activation=tf.abs)
                #variable_summaries_history(mu,self.is_local_net)
            with tf.variable_scope('sigma'):
                sigma = dense(actionNet, num_outputs, None, w_i, None, activation=tf.nn.sigmoid)
                #variable_summaries_history(sigma,self.is_local_net)

            return mu, sigma/10.0
            # return mu, sigma

    # Note: We need 2 return value here: mu & sigma. So it is not suitable to use lazy_property.
    def get_mu_sigma(self, reuse=False):
        # Graph shared with Value Net
        with tf.variable_scope('actor'):
            w_i = tf.random_normal_initializer(0., 1.0)
            b_i = tf.zeros_initializer()
            if self.config.ACTOR_NETWORK_TYPE == 1:
                dense1 = dense(self.state_input, 256, None, w_i, b_i, activation=None)
                # dense1 = tf.contrib.layers.fully_connected(
                #     inputs=tf.contrib.layers.flatten(self.state_input),
                #     num_outputs=256,
                #     scope="dense1")
                variable_summaries_layer(dense1,self.is_local_net)
                if not reuse:
                    tf.contrib.layers.summarize_activation(dense1)
                with tf.variable_scope('mu'):
                    mu = dense(dense1, self.action_dim, None, w_i, None, activation=tf.sin)
                    variable_summaries_history(mu,self.is_local_net)
                with tf.variable_scope('sigma'):
                    sigma = dense(dense1, self.action_dim, None, w_i, None, activation=tf.nn.sigmoid)
                    variable_summaries_history(sigma,self.is_local_net)
            elif self.config.ACTOR_NETWORK_TYPE == 2:
                conv1 = tf.contrib.layers.conv2d(
                    self.state_input, 16, 8, self.config.TEMPORAL_WINDOW, activation_fn=lambda x: (tf.nn.relu(x)), scope="conv1")
                conv2 = tf.contrib.layers.conv2d(
                    conv1, 32, self.config.TEMPORAL_WINDOW, 2, activation_fn=lambda x: (tf.nn.relu(x)), scope="conv2")
                # Fully connected layer
                dense2 = tf.contrib.layers.fully_connected(
                    inputs=tf.contrib.layers.flatten(conv2),
                    num_outputs=256,
                    scope="dense2")
                dense1 = tf.nn.dropout(dense2, 0.9)
                #dense1 = dense(fc1, 200, None, w_i, None, activation=None)
                if not reuse:
                    tf.contrib.layers.summarize_activation(conv1)
                    tf.contrib.layers.summarize_activation(conv2)
                    tf.contrib.layers.summarize_activation(dense1)
                with tf.variable_scope('mu'):
                    mu = dense(dense1, self.action_dim, None, w_i, None, activation=tf.nn.relu)
                    variable_summaries_history(mu,self.is_local_net)
                with tf.variable_scope('sigma'):
                    sigma = dense(dense1, self.action_dim, None, w_i, None, activation=tf.nn.sigmoid)
                    variable_summaries_history(sigma,self.is_local_net)
            elif self.config.ACTOR_NETWORK_TYPE == 4:
                self.dense1 = dense(self.state_input, 200, None, w_i, None, activation=None)
                variable_summaries_layer(self.dense1,self.is_local_net)
                action_dim_half = int(self.action_dim/2)
                #TODO CHANGE REPLAY BUFFER TO OUR NEW IMPLEMENTATION WITH TERMINAL BASED RECALL
                if not reuse:
                    tf.contrib.layers.summarize_activation(self.dense1)
                with tf.variable_scope('muStance'):
                    self.muStance = dense(self.dense1, action_dim_half, None, w_i, None, activation=tf.nn.selu)
                with tf.variable_scope('sigmaStance'):
                    self.sigmaStance = dense(self.dense1, action_dim_half, None, w_i, None, activation=tf.nn.sigmoid)
                with tf.variable_scope('muSwing'):
                    self.muSwing = dense(self.dense1, action_dim_half, None, w_i, None, activation=tf.nn.selu)
                with tf.variable_scope('sigmaSwing'):
                    self.sigmaSwing = dense(self.dense1, action_dim_half, None, w_i, None, activation=tf.nn.sigmoid)
                with tf.variable_scope('mu'):
                    mu = tf.concat([self.muStance,self.muSwing],axis=1)
                    variable_summaries_history(mu,self.is_local_net)
                with tf.variable_scope('sigma'):
                    sigma = tf.concat([self.sigmaStance,self.sigmaSwing],axis=1)
                    variable_summaries_history(sigma,self.is_local_net)
            elif self.config.ACTOR_NETWORK_TYPE == 5:
                # This is the first network that will try to implement some notion of :
                # - symmetry. same network used for left and right leg. Only state input changes.
                # - phase based network. two different network groups used for swing and stance.
                # - general pd control.

                isExtensor = np.array(
                #             HF GLU HAB HAD VAS HAM GAS SOL TA
                             [0, 1,  1,  0,  1,  0,  1,  1,  0,
                              0, 1,  1,  0,  1,  0,  1,  1,  0])
                def concat_with_rule(arr1,arr2,rule):
                    i2 = 0
                    i1 = 0
                    arr = []
                    for r in rule:
                        if r :
                            arr.append(arr1[0,i1])
                            i1 += 1
                        else:
                            arr.append(arr2[0,i2])
                            i2 += 1
                    return tf.expand_dims(tf.stack(arr,axis=-1), 0)

                selector = 1
                multiplier = 1
                mu_ext_st,sigma_ext_st = self.general_actor("extensorSt", self.state_dim, np.sum(isExtensor==selector), ext = multiplier, reuse=True)
                multiplier = 0.3
                mu_ext_sw,sigma_ext_sw = self.general_actor("extensorSw", self.state_dim, np.sum(isExtensor==selector), ext = multiplier, reuse=True)
                selector = 0
                multiplier = -0.3
                mu_flex_st,sigma_flex_st =  self.general_actor("flexorSt", self.state_dim, np.sum(isExtensor==selector), ext = multiplier, reuse=True)
                multiplier = -1
                mu_flex_sw,sigma_flex_sw =  self.general_actor("flexorSw", self.state_dim, np.sum(isExtensor==selector), ext = multiplier, reuse=True)

                mu_st = concat_with_rule(mu_ext_st,mu_flex_st,isExtensor)
                sigma_st = concat_with_rule(sigma_ext_st,sigma_flex_st,isExtensor)

                mu_sw = concat_with_rule(mu_ext_sw,mu_flex_sw,isExtensor)
                sigma_sw = concat_with_rule(sigma_ext_sw,sigma_flex_sw,isExtensor)

                mu = tf.concat([mu_st,mu_sw],axis=1)
                sigma = tf.concat([sigma_st,sigma_sw],axis=1)


                return mu, sigma
            else:
                raise ValueError('Network type "{}" not implemented, should be integer'.format(self.config.ACTOR_NETWORK_TYPE))


            # return mu * self.config.ACTION_BOUND[1], sigma + 1e-4
            return mu, sigma + 1e-4
            # return mu, sigma

    @lazy_property
    def a_prob(self):
        with tf.variable_scope('actor'):
            w_i = tf.random_uniform_initializer(0., 0.1)
            b_i = tf.zeros_initializer()
            with tf.variable_scope('dense1'):
                dense1 = dense(self.state_input, 200, None, w_i, b_i, activation=tf.nn.relu6)
            with tf.variable_scope('dense2'):
                dense2 = dense(dense1, self.action_dim, None, w_i, b_i, activation=tf.nn.softmax)
            return dense2

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
    def choose_action(self):
        if self.config.mode == 'discrete':
            return tf.multinomial(tf.log(self.a_prob), 1)[0][0]  # 莫名其妙，不加tf.log可能出现超出action_dim的值
        elif self.config.mode == 'continuous':
            # axis = 0表示只在第0维上squeeze
            #sample_action = tf.nn.sigmoid(self.action_normal_dist.sample(1))# * self.config.ACTION_GAP + self.config.ACTION_BOUND[0]
            sample_action = self.action_normal_dist.sample(1)# * self.config.ACTION_GAP + self.config.ACTION_BOUND[0]
            self.actions = tf.clip_by_value(tf.squeeze(sample_action, axis=0),
                                    self.config.ACTION_BOUND[0],
                                    self.config.ACTION_BOUND[1])[0]
            return self.actions
