# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from myUtils import discount

GLOBAL_RUNNING_R = []  # record the history scores

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


class Worker(object):
    def __init__(self, name, AC, sess, coord, config, mutex, summary_writer , saver = None, env = None):
        self.sess = sess
        self.coord = coord
        self.mutex = mutex
        self.config = config
        if not env :
            raise ValueError('No env passed to worker')
        self.env = env
        #self.env = env.unwrapped # Not clear what this would do, the author said this : 取消-v0的限制，成绩可以很大
        self.name = name
        self.AC = AC
        self.firstRun = True
        self.saver = saver
        self.summary_writer = summary_writer
        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))
        self.state = None
        #self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.config.N_A))
        basal_act = 0.1
        self.ou_noise_neg = NegativeOrnsteinUhlenbeckActionNoise(mu=-1*basal_act+np.zeros(self.config.N_A),sigma=0.05,theta=0.4,x0=-1*basal_act)
        self.ou_noise_pos = NonNegativeOrnsteinUhlenbeckActionNoise(mu=1*basal_act+np.zeros(self.config.N_A),sigma=0.05,theta=0.4,x0=1*basal_act)
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.config.N_A),sigma=0.05,theta=0.4,x0=basal_act)


    def truncateExperience(self, buffer_s, buffer_a, buffer_r, buffer_v):
        FORGET_LAST_ITEM = self.config.FORGET_WINDOW_SIZE # Half a second at 10ms (e.g. 10 frame-skip)
        # If we are done HERE MEANS fell we go back in time. E.g. we forget one second of the
        # But rewards during last seconds is kept.
        # If not enough observation we skip
        truncated_buffer_s = []
        truncated_buffer_a = []
        truncated_buffer_r = []
        truncated_buffer_v = []
        for i in range(len(buffer_r)-FORGET_LAST_ITEM):
            truncated_buffer_s.append(buffer_s[i])
            truncated_buffer_a.append(buffer_a[i])
            truncated_buffer_r.append(buffer_r[i+FORGET_LAST_ITEM])
            truncated_buffer_v.append(buffer_r[v+FORGET_LAST_ITEM])
        if len(truncated_buffer_r) == 0:
            print("We skip this round because not enough observation in memory")
            return buffer_s,buffer_a,buffer_r,buffer_v
        else:
            if FORGET_LAST_ITEM is not 0 :
                print("Skipping {} out of {} observation".format(FORGET_LAST_ITEM,len(buffer_r)))
            buffer_s = truncated_buffer_s
            buffer_a = truncated_buffer_a
            buffer_r = truncated_buffer_r
            buffer_v = truncated_buffer_v
            return buffer_s,buffer_a,buffer_r,buffer_v


    def train(self,rollout,bootstrap_value):
        gamma = self.config.GAMMA

        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        feed_dict = {
            self.AC.state_input: np.array([np.array(xi) for xi in observations[1:]]),
            self.AC.action_input: np.array([np.array(xi) for xi in actions[1:]]),
            self.AC.action_input_old: np.array([np.array(xi) for xi in actions[:-1]]),
            self.AC.value_target: discounted_rewards[1:,np.newaxis],
            self.AC.advantages: advantages[1:,np.newaxis]
        }
        # import ipdb; ipdb.set_trace()
        # sess = tf.InteractiveSession()
        # mu,sigma,action_input,TD_loss = self.sess.run([self.AC.mu,self.AC.sigma,self.AC.action_input,self.AC.TD_loss], feed_dict=feed_dict)
        # action_normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        # _,summary_episode = self.sess.run([self.AC.push_params,self.AC.summaries], feed_dict=feed_dict)
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
        total_step = 1
        ep_r = 0
        total_step_episode = 1
        episode_buffer, buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], [], []
        while not self.coord.should_stop() and self.config.GLOBAL_EP < self.config.MAX_GLOBAL_EP:
            self.state = np.stack([self.env.reset()] * self.config.TEMPORAL_WINDOW, axis=1)
            self.state = (self.state-self.config.NORMALIZE_MEAN)/self.config.NORMALIZE_STD;
            self.ou_noise.reset()
            while True:
                #if self.name == 'W_0':
                #    self.env.render(mode="human")


                a,v = self.sess.run([self.AC.choose_action,self.AC.value_get_current],feed_dict={self.AC.state_input: self.state[np.newaxis, :]})
                noise_rate = 1.0
                decay = lambda decay,epoch: 1.0/(1.0 + decay*epoch)

                a = self.sess.run(self.AC.choose_action,feed_dict={self.AC.state_input: self.state[np.newaxis, :]})
                a += self.ou_noise()
                #self.summary_writer.add_summary(summ, global_step=total_step)

                s_, r, done, info = self.env.step(a)
                if((np.isnan(a)).any()):
                    import ipdb; ipdb.set_trace()
                # This can happen when webots crashes. We don't get the done info therefore the first time we
                # step we get an empty vector because webots wants to sync with us.
                # If this happens we could force the system to done, and break. or just get next observation
                # And continue like if no terminal state was reached.
                # If webots crashes, it has some chances of being because the simulation entered in a non cool state.
                while len(s_) == 0:
                    s_, r, done, info = self.env.step(a)

                s_ = (s_-self.config.NORMALIZE_MEAN)/self.config.NORMALIZE_STD;
                try:
                    s_ = np.append(self.state[:,1:], np.expand_dims(s_, 1), axis=1)
                except:
                    import ipdb; ipdb.set_trace()

                ep_r += r/self.config.REWARD_SCALING
                buffer_s.append(self.state)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_v.append(v[0,0])
                episode_buffer.append([self.state,a,r,s_,done,v[0,0]])

                self.state = s_

                total_step += 1
                total_step_episode += 1

                if done and self.config.FORGET_WINDOW_SIZE is not 0:
                    buffer_s, buffer_a, buffer_r, buffer_v = self.truncateExperience(buffer_s, buffer_a, buffer_r, buffer_v)

                # update global and assign to local net
                if total_step_episode % self.config.UPDATE_GLOBAL_ITER == 0 or (done and total_step_episode > self.config.UPDATE_GLOBAL_ITER):
                    ep_rn = float(ep_r/total_step_episode*1000)
                    v_s_ = 0 if done else self.sess.run(self.AC.value_get_current, {self.AC.state_input: s_[np.newaxis, :]})[0, 0]
                    # buffer_v_target = []
                    # for r in buffer_r[::-1]:  # reverse buffer r
                    #     v_s_ = r + self.config.GAMMA * v_s_
                    #     buffer_v_target.append([v_s_])
                    # buffer_v_target.reverse()
                    summary_episode = self.train(episode_buffer,v_s_)

                    if summary_episode is not None:
                        summary_outside=tf.Summary()
                        summary_outside.value.add(simple_value=ep_r,tag="eval/episode_reward")
                        summary_outside.value.add(simple_value=total_step_episode,tag="eval/episode_length")
                        summary_outside.value.add(simple_value=ep_rn,tag="eval/episode_reward_normalize")
                        self.summary_writer.add_summary(summary_episode, self.config.GLOBAL_EP)
                        self.summary_writer.add_summary(summary_outside, self.config.GLOBAL_EP)
                        self.summary_writer.flush()
                    # if self.name == 'W_0':
                    #     _,summary_episode = self.sess.run([self.AC.push_params,self.AC.summaries], feed_dict={
                    #         self.AC.state_input: buffer_s,
                    #         self.AC.action_input: buffer_a,
                    #         self.AC.value_target: buffer_v_target,
                    #         })
                    #     summary_outside=tf.Summary()
                    #     summary_outside.value.add(simple_value=ep_r,tag="eval/episode_reward")
                    #     summary_outside.value.add(simple_value=total_step_episode,tag="eval/episode_length")
                    #     summary_outside.value.add(simple_value=ep_rn,tag="eval/episode_reward_normalize")
                    #     self.summary_writer.add_summary(summary_episode, self.config.GLOBAL_EP)
                    #     self.summary_writer.add_summary(summary_outside, self.config.GLOBAL_EP)
                    #     self.summary_writer.flush()
                    #     self.sess.run(self.AC.pull_actor_params)
                    # else:
                    #     self.sess.run([self.AC.push_params], feed_dict={
                    #         self.AC.state_input: buffer_s,
                    #         self.AC.action_input: buffer_a,
                    #         self.AC.value_target: buffer_v_target,
                    #         })
                    #     self.sess.run(self.AC.pull_actor_params)
                    self.mutex.acquire()
                    if self.config.mode == 'discrete':
                        GLOBAL_RUNNING_R.append(ep_r if not GLOBAL_RUNNING_R else 0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    elif self.config.mode == 'continuous':
                        GLOBAL_RUNNING_R.append(ep_r if not GLOBAL_RUNNING_R else 0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print('{:4} Ep: {:4}, Rew: {}, Rew/(per1000step): {}   GLOBAL_RUNNING_R: {}   GLOBAL_STEP: {}'\
                        .format(self.name, self.config.GLOBAL_EP, ep_r, ep_rn, float(GLOBAL_RUNNING_R[-1]), total_step))
                    if self.saver is not None:
                        self.saver.save(self.sess, self.checkpoint_path, global_step=self.config.GLOBAL_EP)
                    self.config.GLOBAL_EP += 1
                    ep_r = 0
                    total_step_episode = 0
                    episode_buffer, buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], [], []
                    self.mutex.release()
                    break
                if done:
                    print("some experience skipped")
                    ep_r = 0
                    total_step_episode = 0
                    episode_buffer, buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], [], []
                    break
