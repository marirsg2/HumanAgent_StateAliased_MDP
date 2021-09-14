# Assume P, r, theta
# P is state transition function giving P(s' | a, s)
# r is a reward function R (s, a, s') or R(s, a)
# theta is threshold > 0 
import numpy as np 
from src.Combined_loss.config import COMPUTATIONAL_TOLERANCE


# returns pi[s], V[s]

# Data structure - V_k[s] is value function at iteration k.

from numpy import linalg as LA

def get_Linf_Norm(newV, V):
	newV = np.array(newV)
	V = np.array(V)

	return LA.norm(newV-V, np.inf)



def ValueIteration(state_set, action_set, R, P, epsilon=0.2, gamma=0.9, ITERATION_CUTOFF=100, d_start=None):
	# probability of starting from a state is d_start

	policy, V_k, V_k_1, k = _ValueIteration(state_set, action_set, R, P, epsilon, gamma, ITERATION_CUTOFF)
	
	V_diff_norm = get_Linf_Norm(V_k, V_k_1) + COMPUTATIONAL_TOLERANCE
	delta_bound = V_diff_norm *(gamma / (1-gamma)) #its 2*eps*gamma/(1-gamma)

	minV = V_k_1 - delta_bound
	maxV = V_k_1 + delta_bound

	return {"minV" : minV, "maxV" : maxV, "V" : V_k_1, "delta" : delta_bound, "policy" : policy, "iteration-k" : k}



def _ValueIteration(state_set, action_set, R, P, epsilon, gamma, ITERATION_CUTOFF) :
	S_size = len(state_set)
	A_size = len(action_set)
	
	V = np.zeros((S_size))
	newV = np.zeros((S_size))

	policy = np.zeros((S_size))

	for k in range(0, ITERATION_CUTOFF):
		newV = np.zeros((V.shape))

		for s in state_set : 

			best_a = [] 

			for a in action_set : 
				vals_sum = R[s,a] 
				for s_ in state_set : 
					# vals_sum += P[s,a,s_]* ( R[s,a,s_] + gamma*V[s_] ) 
					vals_sum += P[s,a,s_]* (gamma*V[s_]) 

				best_a.append(vals_sum)

			newV[s] = max(best_a)

		contFlag = False
		# for s in state_set : 
		# 	if newV[s] - V[s] > epsilon : 
		# 		contFlag = True
		# 		break

		if get_Linf_Norm(newV, V) > epsilon : 
			contFlag = True

		if not contFlag : 
			policy = np.zeros((S_size))
			for s in state_set : 
				best_vals = []
				for a in action_set : 
					action_sum = R[s,a]
					for s_ in state_set : 
						# action_sum += P[s,a,s_]*(R[s,a,s_] + gamma*V[s_])
						action_sum += P[s,a,s_]* (gamma*V[s_])
					best_vals.append([action_sum, a])


				best_action = best_vals[0]
				for bv in best_vals : 
					if (best_action[0] < bv[0]) : 
						best_action = bv

				policy[s] = best_action[1]
			return (policy, V, newV, k)
		V = newV
	return (policy, V, newV, ITERATION_CUTOFF)

						

if __name__ == "__main__" :
	from pprint  import pprint 

	ITERATION_CUTOFF = 100


	state_set = [0,1,2]
	action_set = [0,1]

	# P = np.array((S_size, A_size, S_size))
	# r = np.array((S_size, A_size, S_size))
	theta = 0.3
	gamma = 0.9

	P = np.array([
		[[1, 0, 0], [0, 1, 0]],
		[[1, 0, 0], [0, 0, 1]],
		[[0, 1, 0], [0, 0, 1]]
		]
		)
	# sas style
	# R = np.array([
	# 	[[0,0,1], [0,0,1]],
	# 	[[0,0,1], [0,0,1]],
	# 	[[0,0,1], [0,0,1]]
	# 	]
	# 	)

	# sa style
	R = np.array([
		[0, 0],
		[0, 1],
		[0, 1]
		])

	result = ValueIteration(state_set, action_set, R, P, epsilon=0.001, gamma=0.9, ITERATION_CUTOFF=1000, d_start=None)
	pprint (result)

