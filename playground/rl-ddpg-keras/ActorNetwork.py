import numpy as np
import math
from keras.initializers import random_normal, identity, glorot_normal, lecun_uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Lambda, CuDNNGRU
from keras.optimizers import Adam
import keras.regularizers as regularizers
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, action_bound, BATCH_SIZE, TAU, LEARNING_RATE):
        self.s_dim = state_size
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE,epsilon=0.1).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        #h0 = CuDNNGRU(HIDDEN1_UNITS, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros')(S)
        h0 = Dense(HIDDEN1_UNITS, activation='selu', kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001))(S)
        h1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001))(h0)

        #Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        #Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        #Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        #V = merge([Steering,Acceleration,Brake],mode='concat')
        # TODO: NOTE: WE COULD USE TANH AND BOUND IT TO -BOUND + BOUND. Here its sigmoid activation
        V = Dense(action_dim,activation='sigmoid', kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(h1)
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S
