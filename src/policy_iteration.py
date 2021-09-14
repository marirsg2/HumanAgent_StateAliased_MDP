# value iteration, while max delta v(s) < epsilon, update policy
import numpy as np 
import torch
import random
import math

def policy_improvement(env, seed=1) :
    np.random.seed(seed)

    NUM_STATES = env.NUM_STATES
    NUM_ACTIONS = env.NUM_ACTIONS
    EPSILON = 1E-10
    gamma = 0.5


    curr_policy = np.zeros((NUM_STATES, NUM_ACTIONS))



    for ix in range(NUM_STATES) :
        action = np.random.randint(0,NUM_ACTIONS)
        curr_policy[ix,action] = 1

    curr_policy = np.zeros((NUM_STATES, NUM_ACTIONS))
    curr_policy[:, 2] = 1

    value_vector = np.zeros(NUM_STATES)
    delta_value = np.ones(NUM_STATES)*EPSILON


    # Transition function.
    # SAS2_3d_matx = np.zeros((NUM_STATES,NUM_ACTIONS,NUM_STATES)) #state, action, resultant state 3d matrix
    SAS2_3d_matx = env.compute_3d_sas2_xition_matrix().numpy()



    # 2d reward matrix.
    # SAr_matx = np.zeros((NUM_STATES,NUM_ACTIONS))
    SAr_matx = env.get_reward_vector().numpy()



    #policy iteration
    while np.amax(delta_value) >= EPSILON:
        prev_value_vec = np.copy(value_vector)
        for state_idx in range(NUM_STATES):
            state_filter = np.zeros(NUM_STATES)
            state_filter[state_idx] = 1
            state_filter = state_filter.T #needed for subsequent lin alg op
            reward_vector = np.matmul(state_filter,SAr_matx)
            xition_likelihoods = SAS2_3d_matx[state_idx] #AxS' matrix
            chosen_action_idx = 0
            max_val = -1*math.inf

            ff = list(range(NUM_ACTIONS))
            random.shuffle(ff)
            for action_idx in ff:
                temp_policy = np.copy(curr_policy)
                temp_policy[state_idx] *= 0 #reset
                temp_policy[state_idx][action_idx] = 1.0

                resultant_policy = np.matmul(state_filter ,temp_policy)
                reward_component = np.sum(resultant_policy*reward_vector)
                value_component = np.sum(gamma*np.matmul(resultant_policy,xition_likelihoods)*prev_value_vec)
                value_with_action = reward_component + value_component
                if value_with_action > max_val:
                    chosen_action_idx = action_idx
                    max_val = value_with_action
            #end for loop through actions
            value_vector[state_idx] = max_val
            curr_policy[state_idx] *= 0
            curr_policy[state_idx][chosen_action_idx] = 1
        #end for loop through the states

        delta_value = abs(np.sum(value_vector-prev_value_vec))

    return curr_policy, value_vector
