# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import functools
from scipy import signal


class episodeBuffer:
    s                        = 0
    a_without_noise          = 1
    r                        = 2
    s_next                   = 3
    done                     = 4
    v                        = 5
    a                        = 6
    a_best                   = 7

def getActionList(_buffer): 
    return np.unique(np.array([s[episodeBuffer.a].tolist() for s in _buffer]),axis=0)

def getNonZeroActionList(_buffer):
    action_list = getActionList(_buffer)
    if(len(action_list) == 1):
        return action_list[-1]
    return action_list[np.where(action_list.sum(axis=1) != 0)[0][:]]

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

   @staticmethod
   def ify(text,c):
       return "{}{}{}".format(c,text,color.END)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=-0.05, sigma=0.05, theta=.4, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NegativeOrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=-0.05, sigma=0.05, theta=.4, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        x = np.clip(x,-1,0)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NonNegativeOrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0.05, sigma=0.05, theta=.4, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        x = np.clip(x,0,1)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


decay = lambda decay,epoch: 1.0/(1.0 + decay*epoch) # decay example function, to be put in myUtils

class Explorator:
    # As the network tends to a_best => a_best - a becomes smaller and smaller.
    # The convergence criterion is a threshold on || a_best - a || < epsilon
    # Anyway if you don't do anything then since the difference is becoming small
    # It means that we are not exploring anymore.
    # When this happens it means that we found a local optima for our solution.
    # What we can do then is: update the noise generator so that the probability of looking for solution
    # around a_best is reduced. And start with a new network. But how do we initialize it ?
    # We can initialize it to the closest behavior we want to achieve. So we have something being able to tell us
    # How good we would be on those different behavior. This means that the a sort of value function has to be learned
    # Somehow so that we can evaluate the solution without "running them".
    # That's it we learned a behavior. We can then do that again and again and again.
    # Now imagine we learned

    # We can randomly select an EXPLORING_BEST trial or an exploration trial.
    # We do so by randomly deciding whether the action taken is taken around the optimal of the best solution
    # Or around the current output of the network.


    # TODO : We symmetry in action output to generate better actor.
    # We want to exploit symmetry.
    # For that we need two things a) the reward and b) a phase estimator.
    # We then reward
    def __init__(self,
            p_exploit_best = 0.1,p_explore_best = 0.1,p_explore_actual = 0.8,
            ou_mu = 0.0,ou_sigma = 0.5,ou_theta = 0.4,ou_x0 = 0.0,
            dim = 10, initialEpisode = 0,
            decay_factor = 1e-3 # In 1000 episodes noise is reduced by half.
            ):
        self.dim = dim
        self.EXPLOITING_BEST  = 0
        self.EXPLORING_BEST = 1
        self.EXPLORING_ACTUAL      = 2
        self.FIRST_RUN = 3
        self.explorationTypeStr = ['EXPLOITING_BEST', 'EXPLORING_BEST', 'EXPLORING_ACTUAL','FIRST_RUN']
        self.explorationType = 0
        self.noiseProb = [p_exploit_best, p_explore_best, p_explore_actual]
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=ou_mu*np.ones(dim),sigma=ou_sigma,theta=ou_theta,x0=ou_x0)
        self.ou_noise.reset()
        self.noiseUpdatedAtTimeStep = 0
        self.runSplittingFactor = 2
        self.previousSession = initialEpisode != 0
        self.timesNoiseChanged = 0
        self.episode = initialEpisode-1       # Because we will run chooseNoiseType at every start
        self.initEpisode = initialEpisode     #  of episode which increments the counter by 1
        self.noiseDecay = decay_factor
        self.firstRun = False
    def _getNoiseType(self):
        p_exploit_best = self.noiseProb[0]
        p_explore_best = self.noiseProb[1]
        p_explore_actual = self.noiseProb[2]
        return np.random.choice(3,1,p=[p_exploit_best, p_explore_best, p_explore_actual])[0]
    def getNoiseTypeStr(self):
        return self.explorationTypeStr[self.explorationType-self.timesNoiseChanged]
    def updateNoiseType(self,total_step_trial, longest_trial):
        self.timesNoiseChanged = self.timesNoiseChanged+self._updateNoiseType(total_step_trial, longest_trial)
    def _updateNoiseType(self, total_step_trial, longest_trial):
        # This method is used to updates noise type if criterion reached
        # The idea is that if a solution is in exploitation phase for a certain period of the run
        # it can change to exploratory with the same probablitiy as returned by chooseNoiseType.
        # This means that exploitation can become exploration but not the other way around
        if(longest_trial == 0):
            return 0
        if(total_step_trial-self.noiseUpdatedAtTimeStep == int((longest_trial-self.noiseUpdatedAtTimeStep)/self.factor)):
            actualNoiseType = self.explorationType
            self.noiseUpdatedAtTimeStep = total_step_trial
            if(self.explorationType == self.EXPLOITING_BEST or self.explorationType == self.EXPLORING_BEST):
                self.explorationType = self.explorationType + 1
                return 1
        return 0
    def chooseNoiseType(self):
        self.ou_noise.reset() # TODO Check if we uncomment
        self.factor = 1.5
        self.noiseUpdatedAtTimeStep = 0
        if self.firstRun:
            self.explorationType = self.FIRST_RUN
        else :
            self.explorationType = self._getNoiseType()
        self.timesNoiseChanged = 0
        self.episode += 1
    def updateAction(self,a,a_best=[]):
        if len(a_best) == 0:
            a_best = a
            self.explorationType = self.EXPLORING_ACTUAL


        if(self.explorationType == self.EXPLOITING_BEST):
            return a_best
        if(self.explorationType == self.EXPLORING_BEST):
            return a_best + decay(self.noiseDecay,self.episode)*self.ou_noise()
        if(self.explorationType == self.EXPLORING_ACTUAL):
            return a + decay(self.noiseDecay,self.episode)*self.ou_noise()

    def updateActionConstant(self,a_current):
        return a_current + decay(self.noiseDecay,self.episode)*self.ou_noise()

    def recombine(self,a,b,p=0.5):
        recombinationList = np.random.rand(1,a.shape[0])>0.5 # 50% recombination
        _,IDX = np.where(recombinationList==0)
        _,IDX_INV = np.where(recombinationList==1)
        a[IDX]     = 0.0
        b[IDX_INV] = 0.0
        return a + b


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def variable_summaries_layer(var,write):
    if not write:
        return
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tf.contrib.layers.summarize_activation(var)

def variable_summaries(var,write):
    if not write:
        return
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def variable_summaries_scalar(var,write):
    if not write:
        return
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def variable_summaries_history(var,write):
    if not write:
        return
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        #tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))

# 若bias_shape 为 None，表示不使用bias
def resize_conv(inputs, kernel_shape, bias_shape, strides, w_i, b_i=None, activation=tf.nn.relu):
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    target_height = height * strides[1] * 2
    target_width = width * strides[1] * 2
    resized = tf.image.resize_images(inputs,
                                     size=[target_height, target_width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return conv(resized, kernel_shape, bias_shape, strides, w_i, b_i, activation)


# 替代batch norm，更好的消除对比度
def instance_norm(inputs):
    epsilon = 1e-9  # 避免0除数
    # 在 [1, 2]维度（一个feature map）中求其均值&方差
    mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    return tf.div(inputs - mean, tf.sqrt(tf.add(var, epsilon)))


def residual(inputs, kernel_shape, bias_shape, strides, w_i, b_i=None):
    tmp = conv(inputs, kernel_shape, bias_shape, strides, w_i, b_i)
    with tf.variable_scope('residual'):
        return inputs + conv(tmp, kernel_shape, bias_shape, strides, w_i, b_i)


def conv(inputs, kernel_shape, bias_shape, strides, w_i, b_i=None, activation=tf.nn.relu):
    # 使用tf.layers
    # relu1 = tf.layers.conv2d(input_imgs, filters=24, kernel_size=[5, 5], strides=[2, 2],
    #                          padding='SAME', activation=tf.nn.relu,
    #                          kernel_initializer=w_i, bias_initializer=b_i)
    weights = tf.get_variable('weights', shape=kernel_shape, initializer=w_i)
    conv = tf.nn.conv2d(inputs, weights, strides=strides, padding='SAME')
    if bias_shape is not None:
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        return activation(conv + biases) if activation is not None else conv + biases
    return activation(conv) if activation is not None else conv


# 默认有bias，激活函数为relu
def noisy_dense(inputs, units, bias_shape, c_names, w_i, b_i=None, activation=tf.nn.relu, noisy_distribution='factorised'):
    def f(e_list):
        return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))
    # 使用tf.layers，注意：先flatten
    # dense1 = tf.layers.dense(tf.contrib.layers.flatten(relu5), activation=tf.nn.relu, units=50)
    if not isinstance(inputs, ops.Tensor):
        inputs = ops.convert_to_tensor(inputs, dtype='float')
        # dim_list = inputs.get_shape().as_list()
        # flatten_shape = dim_list[1] if len(dim_list) <= 2 else reduce(lambda x, y: x * y, dim_list[1:])
        # reshaped = tf.reshape(inputs, [dim_list[0], flatten_shape])
    if len(inputs.shape) > 2:
        inputs = tf.contrib.layers.flatten(inputs)
    flatten_shape = inputs.shape[1]
    weights = tf.get_variable('weights', shape=[flatten_shape, units], initializer=w_i)
    w_noise = tf.get_variable('w_noise', [flatten_shape, units], initializer=w_i, collections=c_names)
    if noisy_distribution == 'independent':
        weights += tf.multiply(tf.random_normal(shape=w_noise.shape), w_noise)
    elif noisy_distribution == 'factorised':
        noise_1 = f(tf.random_normal(tf.TensorShape([flatten_shape, 1]), dtype=tf.float32))  # 注意是列向量形式，方便矩阵乘法
        noise_2 = f(tf.random_normal(tf.TensorShape([1, units]), dtype=tf.float32))
        weights += tf.multiply(noise_1 * noise_2, w_noise)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
        assert bias_shape[0] == units
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        b_noise = tf.get_variable('b_noise', [1, units], initializer=b_i, collections=c_names)
        if noisy_distribution == 'independent':
            biases += tf.multiply(tf.random_normal(shape=b_noise.shape), b_noise)
        elif noisy_distribution == 'factorised':
            biases += tf.multiply(noise_2, b_noise)
        return activation(dense + biases) if activation is not None else dense + biases
    return activation(dense) if activation is not None else dense


# 默认有bias，激活函数为relu
def dense(inputs, units, bias_shape, w_i, b_i=None, activation=tf.nn.relu, constraint=None):
    # 使用tf.layers，注意：先flatten
    # dense1 = tf.layers.dense(tf.contrib.layers.flatten(relu5), activation=tf.nn.relu, units=50)
    if not isinstance(inputs, ops.Tensor):
        inputs = ops.convert_to_tensor(inputs, dtype='float')
        # dim_list = inputs.get_shape().as_list()
        # flatten_shape = dim_list[1] if len(dim_list) <= 2 else reduce(lambda x, y: x * y, dim_list[1:])
        # reshaped = tf.reshape(inputs, [dim_list[0], flatten_shape])
    if len(inputs.shape) > 2:
        inputs = tf.contrib.layers.flatten(inputs)
    flatten_shape = inputs.shape[1]
    weights = tf.get_variable('weights', shape=[flatten_shape, units], initializer=w_i, constraint=constraint)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
        assert bias_shape[0] == units
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        return activation(dense + biases) if activation is not None else dense + biases
    return activation(dense) if activation is not None else dense


def flatten(inputs):
    # 使用tf.layers
    # return tf.contrib.layers.flatten(inputs)
    return tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])
    # flatten = tf.reshape(relu5, [-1, np.prod(relu5.shape.as_list()[1:])])
