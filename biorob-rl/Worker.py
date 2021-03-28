# -*- coding: utf-8 -*-
from __future__ import print_function

import datetime
import tensorflow as tf
import numpy as np
import os
from myUtils import Explorator, discount, color, getNonZeroActionList,episodeBuffer
import copy

GLOBAL_RUNNING_R = []  # record the history scores

import time

class Worker(object):
    def __init__(self, name, AC, sess, coord, config, mutex, summary_writer , saver = None, env = None):
        if not env :
            raise ValueError('No env passed to worker')
        self.sess = sess
        self.coord = coord
        self.mutex = mutex
        self.config = config
        self.env = env
        self.name = name
        self.AC = AC
        self.firstRun = True
        self.saver = saver
        self.summary_writer = summary_writer
        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))
        self.state = None # Current env state
        self.explorator = Explorator(
                        dim                 = self.config.N_A,
                        p_exploit_best      = 0.0,
                        p_explore_best      = 1.0,
                        p_explore_actual    = 0.0,
                        initialEpisode      = self.config.GLOBAL_EP
                        )

    def train(self,rollout_list,bootstrap_value):
        # Randomize rollout
        N_obs = len(rollout_list)
        rollout = np.array(rollout_list)
        #for i in range(10):
        #    randbuffer=np.random.choice(range(N_obs),N_obs/4, replace=False)
        g_rnd = lambda w: rollout[randbuffer,w]
        g = lambda w: rollout[:,w]

        observations          = g(episodeBuffer.s)
        actions_without_noise = g(episodeBuffer.a_without_noise)
        rewards               = g(episodeBuffer.r)
        next_observations     = g(episodeBuffer.s_next)
        actions               = g(episodeBuffer.a) # actions + noise
        actions_best          = g(episodeBuffer.a_best) # actions + noise
        values                = g(episodeBuffer.v)

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        gamma = self.config.GAMMA
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        feed_dict = {
            self.AC.state_input: np.array([np.array(xi) for xi in observations[0:]]),
            self.AC.action_input: np.array([np.array(xi) for xi in actions_without_noise[0:]]),
            self.AC.action_input_target: np.array([np.array(xi) for xi in actions_best[0:]]),
            self.AC.value_target: discounted_rewards[0:,np.newaxis],
            self.AC.advantages: advantages[0:,np.newaxis]
        }

        # Good place to debug network
        #
        # import ipdb; ipdb.set_trace()
        # sess = tf.InteractiveSession()
        # mu,action_input,TD_loss = self.sess.run([self.AC.actions,self.AC.action_input,self.AC.TD_loss], feed_dict=feed_dict)
        summary_episode = None
        if self.name == 'W_0':
            try:
                _,summary_episode = self.sess.run([self.AC.push_params,self.AC.summaries], feed_dict=feed_dict)
            except:
                import ipdb; ipdb.set_trace()
        else:
            self.sess.run([self.AC.push_params], feed_dict=feed_dict)
        self.sess.run(self.AC.pull_params)
        return summary_episode


    def work(self):

        split_symmetry = lambda a_current: ([a_current[0]], a_current[1:7:2], a_current[2:7:2], a_current[7:12], a_current[12:])
        combine_symmetric = lambda kref,g,a : np.array(kref+np.insert(g, np.arange(len(g)),g).tolist()+a.tolist()+a.tolist())
        combine_assymetric = lambda kref, g_left, g_right, a_left, a_right : np.array(kref+np.insert(g_right, np.arange(len(g_left)),g_left).tolist()+a_left.tolist()+a_right.tolist())
        total_step = 1
        ep_r = 0
        total_step_episode = 0
        episode_buffer = []

        # We wait for first worker to evaluate best current solution
        if(self.name == "W_0"):
            self.explorator.firstRun = True
            if(self.config.GLOBAL_EP != 0):
                # Previous session found, loading best invid.
                self.AC.globalAC.loadBestInvid()
        else:
            print("Worker {}, is waiting for first worker to finish its initial run".format(self.name))
            initEp = self.config.GLOBAL_EP
            while initEp == self.config.GLOBAL_EP or len(self.AC.globalAC.best_trial_buffer) == 0:
               time.sleep(0.05)
            print("Worker {} is now starting".format(self.name))

        while not self.coord.should_stop() and self.config.GLOBAL_EP < self.config.MAX_GLOBAL_EP:
            # We stack initial state to fill up the temporal window
            #  this has effect only if temporal_window > 1
            self.state = np.stack([self.env.reset()] * self.config.TEMPORAL_WINDOW, axis=1)
            self.state = (self.state-self.config.NORMALIZE_MEAN)/self.config.NORMALIZE_STD;

            # We get best trial from globalAC
            #  we use mutex in order to ensure that no other thread is writing
            #  this ressource at the same time
            self.mutex.acquire()
            best_trial_buffer = copy.deepcopy(self.AC.globalAC.best_trial_buffer)
            best_trial_fitness = self.AC.globalAC.best_trial_fitness
            self.mutex.release()
            # [NEW: ConstantExplorator] 
            # TODO
            # Since we are looking for a constant modulator then
            # the same actor shall be used all over the run. 
            # only two types of constant are accepted : 0 meaning its the normal capacity of the low level control
            #                                           1 meaning its the new behavior. 
            # Here we assume that we start with new behavior and that at one point we need to switch. 
            # Any actor can have maximum 3 different values during the whole run (the 0, the actual best, and the exploratory one)
            # If 3 different values exists we either 
            # - select the last one all the time and compare it to the previous one
            # - we do a recombination of the different values and test them all. 
            #   if we do the recombination approach then there are two approaches 
            #   - we dont store anything and at every run do a recombinaison 
            #     this means that we might loose interesting values that where already found in an other good run 
            #   - we store the K best actors and do recombinaison between them. 

            best_trial = len(best_trial_buffer)
            kref_noise, gamma_noise, _, alpha_noise,_  = split_symmetry(self.explorator.ou_noise())
            symmetric_noise = combine_symmetric(kref_noise,gamma_noise,alpha_noise)
            if(best_trial > 0):
                a_current = copy.deepcopy(best_trial_buffer[-1][episodeBuffer.a])

                # We choose noise type
                self.explorator.chooseNoiseType()


                # Ensure symmetry 
                kref = [a_current[0]]
                gamma_left = a_current[1:7:2]
                gamma_right = a_current[2:7:2]
                alpha_left = a_current[7:12]
                alpha_right = a_current[12:]

                kref, gamma_left, gamma_right, alpha_left, alpha_right = split_symmetry(a_current)

                rnd = np.random.rand()

                gamma = self.explorator.recombine(gamma_left,gamma_right)
                alpha = self.explorator.recombine(alpha_left,alpha_right)
                a_current = combine_symmetric(kref,gamma,alpha)
                

                #if(rnd > 0.8):
                #    current = combine_symmetric(kref,gamma_left,alpha_left)
                #elif(rnd > 0.6):
                #    current = combine_symmetric(kref,gamma_right,alpha_right)
                #elif(rnd > 0.0):
                #else:
                #    current = combine_assymetric(kref,gamma_left, gamma_right,alpha_left, alpha_right)
                #a_current = current


            #print(symmetric_noise)

            # Trial loop
            #  this goes on until we receive a terminal state from the env
            behaviorSelected = False
            while True:
                #
                # Ensure we are synchronized with our client
                #
                if not self.env.model.syncstatus:
                    print("Desync between server and client")
                    # We simply move the episode cursor to match environment state, both on the env
                    # and on the worker by braking the loop which will triger a restart of the client.
                    break
                #
                # Run Network and get actions
                #
                a_without_noise,value = self.sess.run([self.AC.actions,self.AC.value_get_current],feed_dict={self.AC.state_input: self.state[np.newaxis, :]})
                v = value[0,0]
                a = a_without_noise[0]
                #
                # Update action with exploratory noise
                #
                t = 0
                t = 3 if self.explorator.firstRun and self.config.GLOBAL_EP == 0 else t
                t = 1 if self.explorator.firstRun and self.config.GLOBAL_EP != 0 else t
                t = 1 if self.config.MODE == "openloop" else t
                t = 2 if self.config.MODE == "testing"  else t

                #a_best = a if total_step_episode+1 >= best_trial else best_trial_buffer[total_step_episode][episodeBuffer.a]
                #print("---")
                #print(a)
                #print(self.state[-1])
                if t == 0: # [learning] normal state, we explore
                    # [NEW: ConstantExplorator] If we are N steps before falling we get new action.
                    if self.state[-1][1] > 2.0 and not behaviorSelected:
                        behaviorSelected = True
                        if(best_trial > 0):
                            best_dir = self.AC.globalAC.getBestDirection()
                            if(len(best_dir) != 0):
                                k, g1, g2, a1,a2  = split_symmetry(best_dir)
                                gamma = self.explorator.recombine(g1,g2)
                                alpha = self.explorator.recombine(a1,a2)
                                symmetric_dir = combine_symmetric(kref,gamma,alpha)
                                p_exp = 0.5 # TODO THIS Should be global config
                                rnd = np.random.rand()<p_exp
                                if(rnd > 0.7):
                                    self.explorator.explorationType = self.explorator.EXPLOITING_BEST
                                    a_current = a_current + best_dir
                                elif(rnd > 0.4):
                                #    # Would be cool to apply changes in some of the direction only
                                    self.explorator.explorationType = self.explorator.EXPLORING_BEST
                                    a_current = a_current + best_dir*(1+self.explorator.ou_noise())
                                else:
                                    self.explorator.explorationType = self.explorator.EXPLORING_ACTUAL
                                    a_current = a_current + self.explorator.ou_noise()
                            else:
                                a_current += symmetric_noise

                    a = a_current if behaviorSelected else 0*a_current

                    a_best = a # TODO: Change to enable RL learning. 
                               #       We don't learn any actor for now so best action is set to actual action.
                if t == 1: # [testing]  we play the open loop pattern
                    # If we have a best_trial the best action is either 
                    if self.state[-1][1] > 2.0 and not behaviorSelected:
                        behaviorSelected = True

                    a = a_current if behaviorSelected else 0*a_current
                    a_best = a
                if t == 2: # [testing]  we play the network
                    a_best = a
                if t == 3: # [testing]  We don't do anything (check capacity of low level controller)
                    a = 0*a
                    a_best = 0*a

                #print(a)
                #print(a)
                #print("!!!!")
                #
                # Send action to environment
                #
                s_, r, done, info = self.env.step(a)


                #print("{} {}".format(s_[-1],s_[-2]))
                #
                # Append state (if temporal window > 1)
                #
                try:
                    s_ = np.append(self.state[:,1:], np.expand_dims(s_, 1), axis=1) # TODO CHECK
                except:
                    import ipdb; ipdb.set_trace()
                #
                # Store important stuff for learning and log
                #
                ep_r += r/self.config.REWARD_SCALING
                episode_buffer.append([self.state,a_without_noise[0],r,s_,done,v,a,a_best])
                self.state = s_
                total_step += 1
                total_step_episode += 1
                #
                # Update noise type:
                #
                #     this allows to increase "exploration" as
                #     the episode goes one. Update can only go from less to
                #     more exploration
                #
                #self.explorator.updateNoiseType(total_step_episode, best_trial)

                #
                # Terminal state reached
                #
                #
                if done:
                    self.mutex.acquire()
                    #
                    # Update best trial
                    #    if the current trial is better than the best trial
                    #    <=> if the current trial has a bigger fitness
                    #
                    #
                    actual_fitness = r # Last reward before terminal state is reached
                                       # is the fitness (by definition)
                    # Color logging
                    #     to ease checking
                    #
                    actual_fitness_str = "{: 10.4f}".format(actual_fitness)
                    best_fitness_str = "{: 10.4f}".format(self.AC.globalAC.best_trial_fitness)
                    iShallPrint = False
                    #iShallPrint = True
                    if(actual_fitness > self.AC.globalAC.best_trial_fitness):
                        iShallPrint = True
                        actual_fitness_str = color.ify(color.ify("{}  > {}".format(actual_fitness_str,best_fitness_str),color.BOLD),color.GREEN)
                    elif(actual_fitness > 0.85*self.AC.globalAC.best_trial_fitness):
                        iShallPrint = True
                        actual_fitness_str = color.ify(color.ify("{}  <={}".format(actual_fitness_str,best_fitness_str),color.BOLD),color.BLUE)
                    elif(actual_fitness < 0.5*self.AC.globalAC.best_trial_fitness):
                        iShallPrint = False
                        actual_fitness_str = color.ify(color.ify(actual_fitness_str,color.BOLD),color.RED)

                    self.AC.globalAC.updateInvid(episode_buffer,actual_fitness)
                    ep_rn = float(ep_r/total_step_episode*1000)
                    v_s_ = 0 if done else self.sess.run(self.AC.value_get_current, {self.AC.state_input: s_[np.newaxis, :]})[0, 0]

                    #
                    # Learning
                    #
                    #
                    summary_episode = None
                    if self.config.MODE == "training":
                        summary_episode = self.train(episode_buffer,v_s_)
                    #
                    # Summary (for tensorboard)
                    #
                    #
                    if summary_episode is not None:
                        summary_outside=tf.Summary()
                        summary_outside.value.add(simple_value=ep_r,tag="eval/episode_reward")
                        summary_outside.value.add(simple_value=(total_step_episode*self.config.ACTION_REPEAT)/1000.0,tag="eval/episode_duration")
                        summary_outside.value.add(simple_value=actual_fitness,tag="eval/fitness")

                        summary_outside.value.add(simple_value=ep_rn,tag="eval/episode_reward_normalize")
                        self.summary_writer.add_summary(summary_episode, self.config.GLOBAL_EP)
                        self.summary_writer.add_summary(summary_outside, self.config.GLOBAL_EP)
                        self.summary_writer.flush()
                    #
                    # Terminal printing
                    #
                    #
                    GLOBAL_RUNNING_R.append(ep_r if not GLOBAL_RUNNING_R else 0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    if iShallPrint:
                        print('{} {:4} Ep: {:4}   Fit: {}   Step: {: 4}   Rew: {: 10.4f}   G_Rew: {: 10.4f}   G_Step: {:10}   Noise: {},{}'\
                        .format(
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            self.name,
                            self.config.GLOBAL_EP,
                            actual_fitness_str,
                            total_step_episode,
                            ep_r,
                            float(GLOBAL_RUNNING_R[-1]),
                            total_step,
                            self.explorator.getNoiseTypeStr(),
                            self.explorator.timesNoiseChanged
                            ))
                    #
                    # Saving (for tensorboard and future reload)
                    #
                    #
                    if self.saver is not None:
                        self.saver.save(self.sess, self.checkpoint_path, global_step=self.config.GLOBAL_EP)
                    self.config.GLOBAL_EP += 1
                    ep_r = 0
                    total_step_episode = 0
                    episode_buffer = []
                    self.mutex.release()
                    self.explorator.firstRun = False
                    break
