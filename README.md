# Hey there!

This repo is based upon the work done in [learning materials](http://rll.berkeley.edu/deeprlcourse/)
This repo contains binding for Webots in Gym environment and some examples of RL NN Webots. Have fun. 

* [WebotsEnv, webots environment extension for gym]
* [RL-Communication tools, python client/server to communicate between webots gym environment and real webots controller]
* [RL-Learning, cluster ready self behavioral cloning reinforcement learning algorithm (tensorflow)]

## Installation on Linux/OSX
	
1) install miniconda (https://docs.conda.io/en/latest/miniconda.html) or Anaconda (https://www.continuum.io/downloads)

2) install the environment

	```
	conda create -n opensim-rl -c kidzik opensim python=3.6
	source activate opensim-rl
	conda install -c conda-forge lapack git
	pip install osim-rl
	#conda install keras -c conda-forge
	#pip install git+https://github.com/matthiasplappert/keras-rl.git
	pip install zmq
	pip install scipy
	pip install matplotlib
	pip install h5py
	```
	
3a) clonning the repo 

	git clone https://github.com/NickKok/deep-rl-webots.git
	

3b) webotsNoStop 

simple bash script wrapper that restart webots if closed.
check the bash script `biorob-rl-tools/webotsNoStop` for an example.

3c) an example webots connection controller can be found in ./webots_communication_toolbox/controllers

4) run ./INSTALL

## Example 

If you want to use it with [humanWebotsNMM](https://github.com/NickKok/humanWebotsNmm) webots implementation of [libnmm](https://github.com/NickKok/libnmm) then you should 

1) Install humanWebotsNMM 
   follow the guide in the [README.md](https://github.com/NickKok/libnmm). 
2) Add the webots RL connector
   copy the `./webots_communication_toolbox/controllers/regisConnector` folder to `webots/controllers/` of the `humanWebotsNMM` repo 
3) Create some RL enabled world using the template 
   go into `webots/worlds` and run `./createWorlds 3D_RL.wbt 5662 1`. This will create a temporary world file to be used for RL named `tmp_3D_RL_5662.wbt`


Then from your conda environment (e.g. after running `source activate opensim-rl`) you can do things like 

```
from gym.envs import webots                                                                  
import numpy as np                                                                           
env = webots.Regis(action_dim=19,repeat=50,port=5662,world='/path/to/humanWebotsNmm/webots/worlds/tmp_3D_RL_5662.wbt')
env.reset()

for i in range(200):
	observation, reward, done, info = env.step(np.zeros(19))

```

> What is information we get from webots and what effect action have can be fully configured in the ./config/3D_RL/external/brain.yaml file.



Good luck !
