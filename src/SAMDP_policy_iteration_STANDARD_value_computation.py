# value iteration, while max delta v(s) < epsilon, update policy
import numpy as np
import torch

from src.policy_matrix_functions import *
from src.Combined_loss.config import *
import random
import math
import copy





#==========================================================================================
def policy_improvement(all_domain_info,true_policy_value, complexity_weight ,printout_filename,null_actions_enabled=True) :
    print("__file__ = ", __file__)
    EPSILON = 1E-20

    (NUM_STATES,NUM_ACTIONS,NUM_STATES_PER_ROW,classification_matrix,alt_guess_3d_matrix,SAS2_3d_matx,
     rsa_reward_matrix,prob_state,noop_reward_cost_vector,action_mask_invalid_actions) = all_domain_info


    curr_policy = torch.zeros((NUM_STATES, NUM_ACTIONS))
    for ix in range(NUM_STATES) :
        action = random.randint(0,NUM_ACTIONS-1)
        curr_policy[ix,action] = 1

    policy_score = torch.tensor(-math.inf)
    delta_value = torch.tensor(EPSILON)


    #policy iteration
    last_lap = False
    iterations_counter = 0
    with torch.no_grad():
        while delta_value >= EPSILON:
            iterations_counter += 1
            print("Number of full state set iterations =",iterations_counter)
            # if torch.amax(delta_value) < EPSILON:
            #     if last_lap == True:
            #         break #the delta was small, and we went through all states once more to confirm that no more value change could be had.
            #     last_lap = True
            #     # env.translate_and_print_policy(curr_policy)
            # else:
            #     last_lap = False
            prev_policy_score = torch.clone(policy_score)
            randomized_state_indices = list(range(NUM_STATES))
            random.shuffle(randomized_state_indices)
            max_val = -1 * math.inf
            chosen_action_idx = 0
            chosen_state_idx = 0
            for state_idx in randomized_state_indices:
                # if last_lap == True:
                #     if state_idx == 98:
                #         print("catch")
                ff = list(range(NUM_ACTIONS))
                random.shuffle(ff)
                for action_idx in ff:
                    temp_policy = torch.clone(curr_policy)
                    temp_policy[state_idx] *= 0 #reset
                    temp_policy[state_idx][action_idx] = 1.0

                    avg_inverse_value_of_action = iteration_value_computation_inverted_values_sum(prob_state, SAS2_3d_matx, noop_reward_cost_vector,
                                                rsa_reward_matrix, classification_matrix,alt_guess_3d_matrix, action_mask_invalid_actions,
                                                temp_policy,null_actions_enabled=null_actions_enabled)

                    value_of_action = iteration_value_computation(prob_state, SAS2_3d_matx, noop_reward_cost_vector,
                                                rsa_reward_matrix, classification_matrix,alt_guess_3d_matrix, action_mask_invalid_actions,
                                                temp_policy,null_actions_enabled=null_actions_enabled)

                    action_policy_confusion_score = compute_true_policy_confusion_score(classification_matrix,alt_guess_3d_matrix,
                                                                                        temp_policy)

                    # total score is x*(1 +w*(1-c)) c can contribute atmost twice as much to the score when c = 0 and w is 1
                    # total_score = avg_inverse_value_of_action*(1+complexity_weight*(1-action_policy_confusion_score))

                    #avg_inverse_value_of_action will always be [0,1]. In the computation, it is 1/(state_value + 1). So each term is always [0,1]
                    # and so too will be the mean over all terms.
                    #todo NOTE only when the value is known to be 0 or more, can you do the inverse plus 1
                    # total_score = (1-complexity_weight)*avg_inverse_value_of_action  + complexity_weight*action_policy_confusion_score

                    total_score = (1-complexity_weight)*value_of_action  + complexity_weight*action_policy_confusion_score

                    if total_score > max_val:
                        chosen_state_idx = state_idx
                        chosen_action_idx = action_idx
                        max_val = total_score
                #end for loop through actions
            #end for loop through the states
            policy_score = max_val
            curr_policy[chosen_state_idx] *= 0
            curr_policy[chosen_state_idx][chosen_action_idx] = 1
            delta_value = abs(policy_score-prev_policy_score)
    #======end of policy iter search
    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    true_policy_value = compute_expected_policy_value(torch.eye(NUM_STATES),
            rsa_reward_matrix,curr_policy,null_action_likelihood_vector,noop_reward_cost_vector,SAS2_3d_matx,prob_state)

    if null_actions_enabled:
        null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(curr_policy, classification_matrix,alt_guess_3d_matrix)

    expected_policy_value = compute_expected_policy_value(classification_matrix,
            rsa_reward_matrix,curr_policy,null_action_likelihood_vector,noop_reward_cost_vector,SAS2_3d_matx,prob_state)


    return curr_policy, policy_score,classification_matrix,alt_guess_3d_matrix,true_policy_value,expected_policy_value

