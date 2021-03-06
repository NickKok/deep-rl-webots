#! /usr/bin/env python

import unittest
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker
from gym.envs.registration import register

tf.flags.DEFINE_string("model_dir", "/tmp/a3c", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "Breakout-v0", "Name of gym Atari environment, e.g. Breakout-v0")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update")
tf.flags.DEFINE_integer("action_dim", 5, "")
tf.flags.DEFINE_integer("action_repeat", 10, "")
tf.flags.DEFINE_integer("random_seed", 1234, "")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

PORT = 5662

def getEnv(FLAGS,PORT):
    env = None
    try:
        env = gym.make(FLAGS.env)
    except:
        envName = "{}{}".format(FLAGS.env,PORT)
        # Algorithmic
        # ----------------------------------------
        register(
            id=envName,
            entry_point='gym.envs.webots:Regis',
            kwargs=dict(
                action_dim = int(FLAGS.action_dim),
                repeat = int(FLAGS.action_repeat),
                port = PORT
            )
        )
        env = gym.make(envName)
        env.seed(FLAGS.random_seed)
    return env

# Depending on the game we may have a limited action space
_env = getEnv(FLAGS,PORT)

try:
    s_t = _env.reset()
except:
    import ipdb; ipdb.set_trace()
RANDOM_SEED = FLAGS.random_seed


tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)




state_dim = _env.observation_space.shape[0]
action_dim = _env.action_space.shape[0]
action_bound = _env.action_space.high

VALID_ACTIONS = list(range(action_dim))

# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
print("haaaa")
if FLAGS.parallelism:
  NUM_WORKERS = FLAGS.parallelism

print(NUM_WORKERS)
MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
  shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):

  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # Global policy and value nets
  with tf.variable_scope("global") as vs:
    policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS))
    value_net = ValueEstimator(reuse=True)

  # Global step iterator
  global_counter = itertools.count()

  # Create worker graphs
  workers = []
  for worker_id in range(NUM_WORKERS):
    # We only write summaries in one of the workers because they're
    # pretty much identical and writing them on all workers
    # would be a waste of space
    worker_summary_writer = None
    if worker_id == 0:
      worker_summary_writer = summary_writer

    worker = Worker(
      name="worker_{}".format(worker_id),
      env=getEnv(FLAGS,PORT+worker_id+1),
      policy_net=policy_net,
      value_net=value_net,
      global_counter=global_counter,
      discount_factor = 0.99,
      summary_writer=worker_summary_writer,
      max_global_steps=FLAGS.max_global_steps)
    workers.append(worker)

  saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

  # Used to occasionally save videos for our policy net
  # and write episode rewards to Tensorboard
  pe = PolicyMonitor(
    env=_env,
    policy_net=policy_net,
    summary_writer=summary_writer,
    saver=saver)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()

  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
  if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

  # Start worker threads
  worker_threads = []
  for worker in workers:
    worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

  # Start a thread for policy eval task
  monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
  monitor_thread.start()

  # Wait for all workers to finish
  coord.join(worker_threads)
