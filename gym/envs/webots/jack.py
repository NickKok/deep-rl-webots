"""
Gym Environment for the Jack (old Regis) robot of the
Webots simulator (www.cyberbotics)

see https://bitbucket.org/efx/libnmm for details

Author: Florin Dzeladini
"""
import numpy as np
from gym import utils
from gym.envs.webots import webots_env
import matplotlib.pyplot as plt
import pprint as pp

acc = []
acc2 = []


class Jack(webots_env.WebotsEnv):
    def __init__(self, action_dim = 11, repeat = 4, port = 5562):
        self.waitForRight = False
        self.waitForLeft = False
        self.timeLastTouchDown = 0.0
        self.updateType = 0 # Based on frameskip
        self.done = True
        self.rews = None
        if repeat < 0:
            self.updateType = 1 # Until touchdown
        self.state = dict(
            current = dict(
                visualCortexMap = None,
                lastExtraInfo = None,
                timeInCycleLeft = 0,
                timeInCycleRight = 0,
                lastCycleDurationLeft = 1.0,
                lastCycleDurationRight = 1.0,
                leftContact = 0,
                rightContact = 0,
                stepNumber = 0,
                time = 0,
                averageSpeed = 1.3, # Not to far from real one so that iterative mean calculation converges faster
                torsoRotationAccumulator = 0,
                rewardStepper = 0
            ),
            combined = dict(
                lastDuration = 1.0,
                durations = [1.0],
                averageSpeeds = [1.3],
                doneTimes = 0
            )
        )
        #action_dim = 22
        REGIS_CFG = {
            'obs_dim' : 7, # TODO Isn't this filled automatically with the state_vector method ?
            #'action_space' : np.zeros([action_dim,2])+np.array([-1,1])
            'action_space' : np.zeros([action_dim,2])+np.array([0,1])
        }
        webots_env.WebotsEnv.__init__(self, REGIS_CFG, repeat, port)


    def getPhaseEstimation(self):
        phaseRight = 0.0
        phaseLeft = 0.0
        if(self.state["current"]["lastCycleDurationLeft"] > 0.000001):
            #TODO WEIRD, check
            phaseLeft = self.state["current"]["timeInCycleLeft"]/self.state["current"]["lastCycleDurationLeft"]
        if(self.state["current"]["lastCycleDurationRight"] > 0.000001):
            #TODO WEIRD, check
            phaseRight = self.state["current"]["timeInCycleRight"]/self.state["current"]["lastCycleDurationRight"]
        if(phaseLeft > 1.0):
            phaseLeft = 1.0
        if(phaseRight > 1.0):
            phaseRight = 1.0

        return phaseLeft, phaseRight

    def _step(self,action):
        # # 2) Do simulation
        # # ----------------
        ob = []
        reward = 0
        if self.updateType == 0:
            # Step termination based on frame skip
            if self.model.dataThere:
                self.updateState()
            self.do_simulation(action, self.frame_skip)
            ob = self.state_vector()
            self.updateReward()
            reward = self.getReward()

        if(self.done):
            self.finalize()
        # # 5) Return everything
        return ob, reward, self.done, dict()

    def updateState(self):
        self.state["current"]["time"] += self.frame_skip*self.model.opt['timestep']
        currentSpeed = self.model.data['qvel'][0][2];
        currentAverage = self.state["current"]["averageSpeed"]
        currentIteration = self.state["current"]["time"]/(self.frame_skip*self.model.opt['timestep'])
        self.state["current"]["averageSpeed"] += 1.0/(currentIteration+1.0)*(currentSpeed-currentAverage)


        self.state["current"]["leftContact"] = 0
        self.state["current"]["rightContact"] = 0

        if type(self.state["current"]["lastExtraInfo"]) == np.ndarray:
            if self.model.data['extrainfo'][2] == 1 and self.state["current"]["lastExtraInfo"][2] != self.model.data['extrainfo'][2]:
                self.state["current"]["leftContact"] = 1
                self.state["current"]["lastCycleDurationLeft"] = self.state["current"]["timeInCycleLeft"]
                self.state["current"]["timeInCycleLeft"] = 0
            else:
                self.state["current"]["timeInCycleLeft"] += self.frame_skip*self.model.opt['timestep']

            if self.model.data['extrainfo'][0] == 1 and self.state["current"]["lastExtraInfo"][0] != self.model.data['extrainfo'][0]:
                self.state["current"]["rightContact"] = 1
                self.state["current"]["lastCycleDurationRight"] = self.state["current"]["timeInCycleRight"]
                self.state["current"]["timeInCycleRight"] = 0
            else:
                self.state["current"]["timeInCycleRight"] += self.frame_skip*self.model.opt['timestep']


        if self.state["current"]["leftContact"] == 1:
            self.state["current"]["stepNumber"] += 1


        self.state["current"]["lastExtraInfo"] = self.model.data['extrainfo']

        self.state["current"]["rewardStepper"] = 0

        if self.state["current"]["leftContact"] and self.waitForLeft:
            diff = self.state["current"]["time"]-self.timeLastTouchDown
            #print("Worker:{} Left touched @{} D={}".format(self.model.workerID,self.state["current"]["time"],diff))
            if(diff > 0.4):
                self.state["current"]["rewardStepper"] = 1
            if(diff < 0.4):
                self.state["current"]["rewardStepper"] = -1
            self.timeLastTouchDown = self.state["current"]["time"]

            self.waitForRight = True
            self.waitForLeft = False

        if self.state["current"]["rightContact"] and self.waitForRight:
            diff = self.state["current"]["time"]-self.timeLastTouchDown
            #print("Worker:{} Right touched @{} D={}".format(self.model.workerID,self.state["current"]["time"],diff))
            if(diff > 0.4):
                self.state["current"]["rewardStepper"] = 1
            elif(diff < 0.3):
                self.state["current"]["rewardStepper"] = -1.0
            else:
                self.state["current"]["rewardStepper"] = 0.5
            self.timeLastTouchDown = self.state["current"]["time"]

            self.state["current"]["rewardStepper"] = 1
            self.waitForLeft = True
            self.waitForRight = False

        if not self.waitForLeft and not self.waitForRight:
            if self.state["current"]["leftContact"]:
                self.waitForRight = True
            elif self.state["current"]["rightContact"]:
                self.waitForLeft = True

        self.state["current"]["lastExtraInfo"] = self.model.data['extrainfo']

    def finalize(self):
        self.state["combined"]["averageSpeeds"].append(self.state["current"]["averageSpeed"])
        self.state["combined"]["durations"].append(self.state["current"]["time"])
        self.state["combined"]["doneTimes"] += 1
        self.state["combined"]["lastDuration"] = self.state["current"]["time"]
        self.state["current"]["timeInCycleLeft"] = 0.0
        self.state["current"]["timeInCycleRight"] = 0.0
        self.state["current"]["stepNumber"] = 0.0
        self.state["current"]["time"] = 0.0
        self.timeLastTouchDown = 0;
        self.waitForLeft = False
        self.waitForRight = False


    def updateReward(self):
        [self.rews,self.done] = self.getRewards()

    def getCoM(self,q=None):
        if q is None:
            q = self.model.data['qpos']
        m = np.array(self.model.data['mass'])
        #return np.linalg.norm((q*np.transpose(np.array([m,m,m])))/np.sum(m),axis=0)
        x = (q[:,0].dot(m))/sum(m)
        y = (q[:,1].dot(m))/sum(m)
        z = (q[:,2].dot(m))/sum(m)
        return np.array([x,y,z])

    def getReward(self):
        reward = 0.0

        # OPTIMIZE_FOR = "recoverer"
        OPTIMIZE_FOR = "mujoco-humanoid"
        # Check Regis for the others
        # OPTIMIZE_FOR = "min_speed"
        # OPTIMIZE_FOR = "robustness"

        if(OPTIMIZE_FOR == "mujoco-humanoid"):
            alive_bonus = 5.0
            pos_after = self.model.data['qpos'][0,[0,2]]
            pos_before = self.model.lastData['qpos'][0,[0,2]]
            com_y = self.model.data['qpos'][0][1]
            com = np.linalg.norm((self.model.data['qpos'][:,[1,2]]*np.transpose(np.array([self.model.data['mass'],self.model.data['mass']])))/np.sum(self.model.data['mass']),axis=0)
            com_y = com[0]
            ctrl = self.model.data['ctrl']
            cfrc_ext = self.model.data['extrainfo'][-1] # TODO Implement
            lin_vel_cost = 0.25 * np.linalg.norm(pos_after - pos_before) / self.model.opt['timestep']
            quad_ctrl_cost = 0.1 * np.square(ctrl).sum()
            quad_impact_cost = 0.1*np.square(cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            self.done = bool((com_y < 0.9) or (com_y > 2.0))
        elif(OPTIMIZE_FOR == "recoverer"):
            dt = self.frame_skip*self.model.opt['timestep']
            CoM = self.getCoM(self.model.data['qpos'])
            CoM_old = self.getCoM(self.model.lastData['qpos'])

            CoM_speed = (CoM - CoM_old)/dt
            reward = CoM[1]-abs(CoM_speed[1])-abs(CoM_speed[2])
        else:
            print("Something is definitely wrong, you try to optimizer for '{}' but this does not exist".format(OPTIMIZE_FOR))


        #reward = 0.1*self.state["current"]["time"]*(self.model.data['qvel'][0][2]-minimumToleratedSpeed)
        #- self.model.data['extrainfo'][4] - self.model.data['extrainfo'][5]
        #if self.done:
        #    reward = self.rews["punishmentFall"]

        return reward

    def getRewards(self):
        desiredSpeed = 0.4
        maxTime = 5
        toleranceSpeed = 0.1

        done = False
        _,doneRew = self.rewardLegInFront(self.state["current"]["rightContact"], self.model.data['qpos'][-1][2], self.model.data['qpos'][0][2])
        done = done or doneRew

        _,doneRew = self.rewardLegInFront(self.state["current"]["leftContact"], self.model.data['qpos'][3][2], self.model.data['qpos'][0][2])
        done = done or doneRew

        distanceBetweenTorsoAndLeg = self.model.data['qpos'][0][1]-self.model.data['qpos'][-1][1]
        _, doneRew = self.reward_punishmentFall(distanceBetweenTorsoAndLeg, 0.8, 1.0)
        done = done or doneRew

        _, doneRew = self.rewardDuration(self.state["current"]["time"],maxTime)
        done = done or doneRew

        distanceSinceOrigin = np.linalg.norm(self.model.data['qpos'][0][:]);
        _, doneRew = self.rewardDistance(distanceSinceOrigin,10)
        done = done or doneRew

        return [None, done]

    def rewardDuration(self, duration, maxDuration):
        if duration >= maxDuration:
            return [10.0, True]
        else:
            return [1.0, False]

    def rewardDistance(self, distance, desiredDistance):
        return [distance/desiredDistance, False]

    def rewardLegInFront(self,legContact,legPosition,torsoPosition, consideredInFrontAt = 0.0):
        if(legContact):
            # If leg is in front we return a positive reward
            if legPosition - torsoPosition > consideredInFrontAt:
                return [1.0, False]
            else:
                return [-1.0, False]
        else:
            return [0.0, False]


    def reward_punishmentFall(self,actualHeight,minimalHeight,factor):
        if(actualHeight < minimalHeight):
            return [-1*factor, True]
        else:
            return [0, False]


    def reward_desiredSpeed(self,actualSpeed,desiredSpeed,tolerance):
        # Reward to optimize for a desired speed within a tolerance in speed [m/s]

        # NOTE: If we just return a -1 or 1 and if we start to far from the desired speed
        # We will never understand that he needs to do that.
        # Instead we could remember the minimal achieved speed (at least during x time step)
        # And use it as the next target.
        allSpeeds = np.array(self.state["combined"]["averageSpeeds"])
        closerSpeed = allSpeeds[np.argmin(abs(allSpeeds - desiredSpeed))]
        self.state["combined"]["bestVelocity"] = closerSpeed
        # We wanna do better than closerSpeed which might be very far from desiredSpeed but
        # this allows us to say "good boy" to our robot if speed gets closer
        # This allows us to stay simply in the reward we give but we are smart in the way
        # to choose it
        if actualSpeed - closerSpeed < 0: # We wanna go always slower so we should
            return [1, False]             # Be careful to tell the direction. Increasing speed would require > 0
        else:
            #return [-1*abs(self.model.data['qvel'][0][2]-desiredSpeed)/desiredSpeed, False]
            return [-1, False]


    def state_vector(self):
        leftContact_previous = 0+self.state["current"]["leftContact"]
        rightContact_previous = 0+self.state["current"]["rightContact"]

        phaseLeft,phaseRight = self.getPhaseEstimation()
        #    [0] =  'LEFT_STANCE'
        #    [1] =  'LEFT_SWING'
        #    [2] =  'RIGHT_STANCE'
        #    [3] =  'RIGHT_SWING'
        #    [4] =  'HIP_LEFT'
        #    [5] =  'HIPCOR_LEFT'
        #    [6] =  'KNEE_LEFT'
        #    [7] =  'ANKLE_LEFT'
        #    [8] =  'HIP_RIGHT'
        #    [9] =  'HIPCOR_RIGHT'
        #    [10] = 'KNEE_RIGHT'
        #    [11] = 'ANKLE_RIGHT'
        #    [12] = 'THETA_TRUNK'
        #    [13] = 'ENERGY_DT'
        #    [14] = 'ENERGY_OE_DT'
        #    [15] = 'SELF_CONTACT_COUNTER'
        dt = self.frame_skip*self.model.opt['timestep']
        com = self.getCoM(self.model.data['qpos'])
        com_old = self.getCoM(self.model.lastData['qpos'])
        com_speed = (com-com_old)/dt
        if(len(self.model.lastData['extrainfo']) is not len(self.model.data['extrainfo'])):
            self.model.lastData['extrainfo'] = self.model.data['extrainfo']
        return np.concatenate([
            self.model.data['extrainfo'][[0,2]],
            #[phaseLeft,phaseRight],
            #[-phaseLeft,-phaseRight],
            #(self.model.data['extrainfo'][[4,8,12]]-self.model.lastData['extrainfo'][[4,8,12]])/dt,
            self.model.data['extrainfo'][4:12],
            self.getCoM(self.model.data['qpos']),
            self.getCoM(self.model.data['qpos'])-self.getCoM(self.model.lastData['qpos']),
        ])
