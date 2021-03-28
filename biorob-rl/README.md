# Introduction 

After many tets and trials this is the algorithm selected as the first one in biorob-rl. 

This algorithm is a mixture of different algorithms : 

A policy learning with behavioral cloning. The used behavior is the best individual found so far. 
The key point of our method is two test inviduals with a global fitness. Use exploratory noise. 
Every actor tries to converge to the action of the best inviduals.


The  value function that is used only for states going further than the length of the


# To Do list 

1) Make state vector able to use past experiences. 
2) Make convergence to best actor instantaneous (or very fast). 
3) Make the noise smarter and be able to learn several behavior.
	Noise is exploratory. 
	However adding always the same amount of noise migth prevent the convergence. 
	Indeed as we use behavioral cloning on the action produced by the actor and the real action used by the best invidiual (i.e. with noise)
		We want the solution that tries to converge to the best solution to generate less noise. So that we explore around the current existing solution 

4) Using exploratory path.
	If we know what was the noise added for the best solution we could also produce noise around those trajectories directly. 
	In that case we do make the convergence toward the best actor. And always produce an exploratory noise which is the difference of the current actor output 
	and the noise. So that we always explore around the best invidid but slowly transfer information from the noise (exploration) into the current actor (exploitation)

	
	We could do : 

		best_actions = best_actor_output + best_noise
		actions = actor_output + noise

		=> 

		actions = actor_output + special_noise 


		we want : actions = best_actions 
			actor_output + special_noise = best_actor_output + best_noise
						     = best_actions

			special_noise = (best_actor_output + best_noise - actor_output)*(1 + mu*normal_noise) 

				This means that we are putting noise around the best solution. 


			actor_output + (best_actor_output + best_noise - actor_output)*(1 + mu*normal_noise) 

				This formula is the exploratory path. 


	So we don't only have behavioral cloning but we then do an exploration around the best path found so far. 
	This will converge to some local minima. But this is where the beauty of the method comes into play. 

	We first define a "convergence criterion". Once this convergence criterion has been reached, it means we found a 
	a behavior. What we can then assume is that we are looking for better solution on other places of the state space. 
	So for that we do a special noise which acts on a reduce space (how to define this reduced space is mathematically still not clear but 
	conceptually its ok, we just reduce the probability of generating noise in the region of state space where the previous behavior has been found. 
	
	We also clone the network and start with a new network. The output of this network are the ropes. 

	Now different behavior can be combinaison of other behavior. So when looking for the second behavior it can also be a combinaison of previous behavior. 
	We can therefore randomly "mix" different networks. 

 if a different and better solution exists 

	

			

		special_noise = 


# Examples 



STAIRS
python a3C.py --action-dim=17 --update-batch-iter=150 --action-repeat=50 --critic-lr=0.001 --actor-lr=0.0001 --input-space-type=0 --temporal-window=1 --reward-scaling=1 --actor-network-type=1 --entropy-beta=0.0001 --world=/home/efx/Development/PHD/AiriNew/humanWebotsNmm/controller_current/webots/worlds/3D_RL_slope.wbt --forget-window-size=0




LEARN 3D 
python a3C.py --action-dim=24 --update-batch-iter=500 --action-repeat=10 --critic-lr=0.0001 --actor-lr=0.001 --input-space-type=2 --temporal-window=1 --reward-scaling=1000 --actor-network-type=4 --critic-network-type=3 --entropy-beta=0.001 --forget-window-size=0
