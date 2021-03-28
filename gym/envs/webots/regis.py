"""
Gym Environment for the Regis robot of the
Webots simulator (www.cyberbotics)

see https://bitbucket.org/efx/libnmm for details

Author: Florin Dzeladini
"""
import numpy as np
from gym import utils
from gym.envs.webots import webots_env
import matplotlib.pyplot as plt
import pprint as pp


class Regis(webots_env.WebotsEnv):
    def __init__(self, action_dim = 11, repeat = 4, port = 5562, world = ""):
        self.waitForRight = False
        self.waitForLeft = False
        self.timeLastTouchDown = 0.0
        self.updateType = 0 # Based on frameskip
        self.done = False
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
        self.action_dim = action_dim
        REGIS_CFG = {
            'obs_dim' : 7, # TODO Isn't this filled automatically with the state_vector method ?
            #'action_space' : np.zeros([action_dim,2])+np.array([-1,1])
            'action_space' : np.zeros([action_dim,2])+np.array([0,1])
        }
        if(world == ""):
            raise ValueError("world can not be empty")
        webots_env.WebotsEnv.__init__(self, REGIS_CFG, repeat, port, world)

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

    def step(self,action):
        return self._step(action)

    def _step(self,action):
        # # 2) Do simulation
        # # ----------------
        ob = []
        reward = 0
        if self.updateType == 0:
            # Step termination based on frame skip
            if self.model.dataThere:
                self.updateState()
                reward = self.getReward()
                self.done = self.isDone()
            self.do_simulation(action, self.frame_skip)
            ob = self.state_vector()
            
            if(self.done):
                reward = np.linalg.norm(self.model.data['qpos'][0][:])
        else:
            # Step termination based on event
            if self.state["current"]["time"] == 0:
                if self.model.dataThere:
                    self.updateState()
                self.do_simulation(action, 10)
                ob = self.state_vector()
                self.updateReward()
            else:
                while True:
                    if self.model.dataThere:
                        self.updateState()
                    self.do_one_simulation_step(action)
                    ob = self.state_vector()
                    reward = self.getReward()
                    if self.done or self.state["current"]["leftContact"] == 1 or self.state["current"]["rightContact"] == 1:
                        break

        if(self.done):
            self.finalize()

        return ob, reward, self.done, dict(
                #rewardDistance = CoM[0]
            )

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


    def updateReward(self): # Deprecated 
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

    def isDone(self):
        CoM = self.getCoM(self.model.data['qpos'])
        com_y = CoM[1]
        done = bool((com_y < 0.9) or (com_y > 2.0))
        return done;

    def state_vector(self):
        leftContact_previous = 0+self.state["current"]["leftContact"]
        rightContact_previous = 0+self.state["current"]["rightContact"]

        phaseLeft,phaseRight = self.getPhaseEstimation()

        prediction_error = self.model.data['extrainfo'][-1]
        dt = self.frame_skip*self.model.opt['timestep']
    
        com = self.getCoM(self.model.data['qpos'])

        com_old = self.getCoM(self.model.lastData['qpos'])
        com_speed = (com-com_old)/dt

        distanceSinceOrigin = np.linalg.norm(self.model.data['qpos'][0][:])
        #if(len(self.model.lastData['extrainfo']) is not len(self.model.data['extrainfo'])):
        #    self.model.lastData['extrainfo'] = self.model.data['extrainfo']
        return np.concatenate([
            self.model.data['extrainfo'][0:9],
            #self.model.data['extrainfo'][[0,2]], # Contact info
            #self.getCoM(self.model.data['qpos']),
        ])



    ###################################
    ###################################
    ###################################
    ########### Deprecated ############
    ###################################
    ###################################


    def getVisualCortexRepresentation(self):
        pointsToDraw =  100*((self.model.data['qpos'] - self.model.data['qpos'][0])[:,[1,2]])*[-1,1]
        visualCortexMap = ContinousStateToDiscreMap([50,50],[200,200], [0.15,0.55],pointsToDraw)

        if(self.model.data['extrainfo'][0] == 1):
            visualCortexMap.drawpointNormalized([0.94,0.2],5)
        if(self.model.data['extrainfo'][1] == 1):
            visualCortexMap.drawpointNormalized([0.94,0.1],5)
        if(self.model.data['extrainfo'][2] == 1):
            visualCortexMap.drawpointNormalized([0.94,0.8],5)
        if(self.model.data['extrainfo'][3] == 1):
            visualCortexMap.drawpointNormalized([0.94,0.9],5)



        return visualCortexMap



    def getReward(self):
        reward = 0.0

        #OPTIMIZE_FOR = "mujoco-humanoid"
        #OPTIMIZE_FOR = "robustness"
        # OPTIMIZE_FOR = "min_speed"
        OPTIMIZE_FOR = "faster than my ghost"

        if(OPTIMIZE_FOR == "faster than my ghost"):
            v = 0.5 # [m/s]
            t = self.state["current"]["time"] # [s]
            X = self.getCoM(self.model.data['qpos']) # [m,m,m]
            x = X[2] # position
            y = X[1]
            rewardx = x-v*t
            rewardy = y-1.0
            reward = -(rewardx*rewardx)
            reward = min(1,(max(-1,reward)))
        elif(OPTIMIZE_FOR == "mujoco-humanoid"):
            alive_bonus = 5.0
            CoM = self.getCoM()
            lastCoM = self.getCoM(self.model.lastData['qpos'])
            diffCoM = CoM - lastCoM
            com_y = CoM[1]
            ctrl = self.model.data['ctrl']
            cfrc_ext = self.model.data['extrainfo'][-1] # TODO Implement
            hori_vel_cost = 0.1 * np.linalg.norm(diffCoM[[0,2]]) / self.model.opt['timestep'] # 0.25
            vert_vel_cost = 0.1 * np.linalg.norm(diffCoM[1]) / self.model.opt['timestep'] # 0.25
            vert_vel_cost = max(vert_vel_cost-0.01,0)*10
            quad_ctrl_cost = 0.1 * np.square(ctrl).sum()
            quad_impact_cost = 0.1*np.square(cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            too_low_cost = 0 * max(com_y - 1.0,0.0)
            reward = alive_bonus + 0*hori_vel_cost - (quad_ctrl_cost + quad_impact_cost + too_low_cost + vert_vel_cost)
            #print(" hori_vel_cost {}".format( hori_vel_cost ))
            #print(" alive_bonus {}".format(alive_bonus ))
            #print(" vert_vel_cost {}".format( vert_vel_cost ))
            #print(" quad_ctrl_cost {}".format(quad_ctrl_cost ))
            #print(" quad_impact_cost {}".format(quad_impact_cost ))
            #print(" too_low_cost{}".format(too_low_cost))
        elif(OPTIMIZE_FOR == "robustness"):
            if(self.state["current"]["time"] < 1):
                reward = -self.model.lastData['extrainfo'][-1]
            else:
                reward = -self.model.lastData['extrainfo'][-1]/10
            #reward = self.rews["distanceSinceOrigin"]
        elif(OPTIMIZE_FOR == "min_speed"):
            dt = self.frame_skip*self.model.opt['timestep']
            maximumToleratedSpeed = 1.3
            minimumToleratedSpeed = 0.4
            desiredSpeed = 0.5
            #maximumToleratedSpeed = 1.5
            #minimumToleratedSpeed = 1.2

            com = self.getCoM(self.model.data['qpos'])
            com_old = self.getCoM(self.model.lastData['qpos'])
            com_speed = (com-com_old)/dt

            speed = com_speed[2]

            self.state["current"]["time"]
            if(speed < minimumToleratedSpeed):
                reward = -2
            elif(speed > maximumToleratedSpeed):

                reward = -2
            else:
                reward = 2-np.abs(speed-desiredSpeed)
        else:
            print("Something is definitely wrong, you try to optimizer for '{}' but this does not exist".format(OPTIMIZE_FOR))


        #reward = 0.1*self.state["current"]["time"]*(self.model.data['qvel'][0][2]-minimumToleratedSpeed)
        #- self.model.data['extrainfo'][4] - self.model.data['extrainfo'][5]
        # if self.done:
        #    reward = self.rews["punishmentFall"]

        return reward

    def getRewards(self):
        desiredSpeed = 0.4 # TODO Those limits should be set from a config file not here.
        maxTime = 5 # TODO Those limits should be set from a config file not here.
        toleranceSpeed = 0.1 # TODO Those limits should be set from a config file not here.

        done = False
        [rewardRightContactPosition,doneRew] = self.rewardLegInFront(self.state["current"]["rightContact"], self.model.data['qpos'][-1][2], self.model.data['qpos'][0][2])
        done = done or doneRew
        #print("done:{}".format(doneRew))
        #print("rightContact: {}, reward_RightContactPosition: {}".format(self.state["current"]["rightContact"], rewardRightContactPosition))
        [rewardLeftContactPosition,doneRew] = self.rewardLegInFront(self.state["current"]["leftContact"], self.model.data['qpos'][3][2], self.model.data['qpos'][0][2])
        done = done or doneRew
        #print("done:{}".format(doneRew))
        #print("leftContact: {}, reward_LeftContactPosition: {}".format(self.state["current"]["leftContact"], rewardLeftContactPosition))
        forwardSpeed = self.model.data['qvel'][0][2];

        [rewardSpeed, doneRew] = self.reward_desiredSpeed(forwardSpeed, desiredSpeed, toleranceSpeed)
        done = done or doneRew
        #print("done:{}".format(doneRew))
        #print("speed: {}, reward_speed: {}".format(forwardSpeed,rewardSpeed))
        distanceBetweenTorsoAndLeg = self.model.data['qpos'][0][1]-self.model.data['qpos'][-1][1]
        #[rewardPunishmentFall, doneRew] = self.reward_punishmentFall(distanceBetweenTorsoAndLeg, 0.8, 1.0/(self.state["current"]["stepNumber"]+1.0))
        [rewardPunishmentFall, doneRew] = self.reward_punishmentFall(distanceBetweenTorsoAndLeg, 0.8, 1.0)
        done = done or doneRew
        #print("done:{}".format(doneRew))
        #print("distanceBetweenTorsoAndLeg: {}, reward_punishmentFall: {}".format(distanceBetweenTorsoAndLeg, rewardPunishmentFall))

        [rewardDuration, doneRew] = self.rewardDuration(self.state["current"]["time"],maxTime)
        done = done or doneRew
        #print("done:{}".format(doneRew))
        #print("duration: {}, reward_duration: {}".format(self.state["current"]["time"], rewardDuration))

        distanceSinceOrigin = np.linalg.norm(self.model.data['qpos'][0][:]);
        [rewardDistance, doneRew] = self.rewardDistance(distanceSinceOrigin,10)
        done = done or doneRew
        #print("done:{}".format(doneRew))

        # Print info every x ms
        # if(int(1000*self.state["current"]["time"]) % 500 == 0):
        #     print("rewardDuration: {}".format(rewardDuration))
        #     print("rewardDistance: {}".format(rewardDistance))
        #     print("rewardLeftContactPosition: {}".format(rewardLeftContactPosition))
        #     print("rewardRightContactPosition: {}".format(rewardRightContactPosition))
        #     print("rewardPunishmentFall: {}".format(rewardPunishmentFall))
        #     print("rewardSpeed: {}".format(rewardSpeed))
        #     print("done: {}".format(done))

        return [dict(
                duration = rewardDuration,
                distance = rewardDistance,
                leftContactPosition = rewardLeftContactPosition,
                rightContactPosition = rewardRightContactPosition,
                punishmentFall = rewardPunishmentFall,
                speed = rewardSpeed,
                forwardSpeed = forwardSpeed,
                distanceSinceOrigin = distanceSinceOrigin
            ), done]

    def rewardDuration(self, duration, maxDuration):
        # This follows the same principle than the calculation of the reward associated to speed
        # NOTE ATTENTION !!!
        # Changing the way reward are considered should also change the associated reward
        # of the buffer. Since a positive reward could now mean negative reward
        # We can also drop everything we know whenever we improve
        # But this seems less plausible
        # What could happen instead is that all past information are reevaluated
        durations = np.array(self.state["combined"]["durations"])
        bestDuration = np.max(durations)
        medianDuration = np.median(np.array(self.state["combined"]["durations"]))
        self.state["combined"]["bestDuration"] = bestDuration
        self.state["combined"]["medianDuration"] = medianDuration

        # TODO: NOTE: CHANGED!!!
        # if duration > maxDuration:
        #     return [1.0, True]
        # elif duration > bestDuration:
        #     return [0.5, False]
        # elif duration > medianDuration:
        #     return [0.1, False]
        # else:
        #     return [0.0, False]
        #
        # If this changes every time a better one is done we are not gonna converge.
        # Simply because we did not reinforce enough the best.
        # What we want instead is something that counts the number of time
        # We got that far. And whenever in the last X runs we were good we consider it as
        # learned and increase difficulty.
        takeLastX = 20
        if(takeLastX < len(durations)):
            takeLastX = len(durations)
        unique,counts = np.unique(np.round(durations[len(durations)-takeLastX:],1), return_counts=True)
        #bestStableDuration = unique[np.argmax(counts)]
        takeLastMostRepeteatedX = 2
        if len(counts) < 2:
            takeLastMostRepeteatedX = 1

        bestStableDuration = np.mean(unique[np.argpartition(counts, -takeLastMostRepeteatedX)[-takeLastMostRepeteatedX:]])

        self.state["combined"]["bestStableDuration"] = bestStableDuration
        #if duration > maxDuration:
        #    return [1.0, True]
        #elif duration > bestStableDuration:
        #    return [0.01, False]
        #else:
        #    return [0.0, False]
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