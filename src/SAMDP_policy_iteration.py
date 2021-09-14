# value iteration, while max delta v(s) < epsilon, update policy
from src.policy_matrix_functions import *
from src.Combined_loss.config import *
import random
import math


def policy_improvement(env, complexity_weight ,printout_filename,null_actions_enabled=True) :
    NUM_STATES = env.NUM_STATES
    NUM_ACTIONS = env.NUM_ACTIONS
    EPSILON = 1E-20


    curr_policy = torch.zeros((NUM_STATES, NUM_ACTIONS))
    for ix in range(NUM_STATES) :
        action = random.randint(0,NUM_ACTIONS-1)
        curr_policy[ix,action] = 1

    policy_score = torch.tensor(0)
    delta_value = torch.tensor(EPSILON)

    #classification_matrix
    confusion_potentials_matrix = env.get_confusion_potentials_matrix()
    confusion_normaliz_denom = torch.sum(confusion_potentials_matrix, dim=1)
    confusion_normaliz_denom = confusion_normaliz_denom.view(-1, 1).repeat(1, NUM_STATES)
    classification_matrix = confusion_potentials_matrix/confusion_normaliz_denom
    classification_matrix = classification_matrix


    # Transition function.
    # SAS2_3d_matx = torch.zeros((NUM_STATES,NUM_ACTIONS,NUM_STATES)) #state, action, resultant state 3d matrix
    SAS2_3d_matx = env.compute_3d_sas2_xition_matrix()

    # 2d reward matrix.
    rsa_reward_matrix = env.get_reward_vector(invalid_action_penalty=INVALID_ACTION_PENALTY)
    max_reward_abs = torch.abs(torch.max(rsa_reward_matrix))

    prob_state = torch.ones(NUM_STATES)/NUM_STATES
    noop_reward_cost = NOOP_COST + -1 * max_reward_abs * complexity_weight
    noop_reward_cost_vector = torch.ones(NUM_STATES) * noop_reward_cost

    action_mask_invalid_actions = torch.ones((NUM_STATES, NUM_ACTIONS))
    if REMOVE_INVALID_ACTIONS:
        action_mask_invalid_actions = env.get_mask_for_invalid_actions()

    #policy iteration
    last_lap = False
    iterations_counter = 0
    with torch.no_grad():
        while torch.amax(delta_value) >= EPSILON:
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
            for state_idx in randomized_state_indices:
                # if last_lap == True:
                #     if state_idx == 98:
                #         print("catch")
                chosen_action_idx = 0
                max_val = -1*math.inf
                ff = list(range(NUM_ACTIONS))
                random.shuffle(ff)
                for action_idx in ff:
                    temp_policy = torch.clone(curr_policy)
                    temp_policy[state_idx] *= 0 #reset
                    temp_policy[state_idx][action_idx] = 1.0

                    value_of_action = iteration_value_computation(prob_state, SAS2_3d_matx, noop_reward_cost_vector,
                                                rsa_reward_matrix, classification_matrix, action_mask_invalid_actions,
                                                temp_policy,null_actions_enabled=null_actions_enabled)


                    if value_of_action > max_val:
                        chosen_action_idx = action_idx
                        max_val = value_of_action
                #end for loop through actions
                policy_score = max_val
                curr_policy[state_idx] *= 0
                curr_policy[state_idx][chosen_action_idx] = 1
            #end for loop through the states
            delta_value = abs(torch.sum(policy_score-prev_policy_score))
    #======end of policy iter search
    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    #update rewards for final computation removing any tuning used for search
    rsa_reward_matrix = env.get_reward_vector(invalid_action_penalty=INVALID_ACTION_PENALTY)
    noop_reward_cost_vector = torch.ones(NUM_STATES) * NOOP_COST
    #get the true policy value by using identity matrix as the classification matrix
    true_policy_value = compute_expected_policy_value(torch.eye(NUM_STATES),
            rsa_reward_matrix,curr_policy,null_action_likelihood_vector,noop_reward_cost_vector,SAS2_3d_matx,prob_state)

    if null_actions_enabled:
        null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(curr_policy, classification_matrix)

    expected_policy_value = compute_expected_policy_value(classification_matrix,
            rsa_reward_matrix,curr_policy,null_action_likelihood_vector,noop_reward_cost_vector,SAS2_3d_matx,prob_state)


    return curr_policy, policy_score,classification_matrix,true_policy_value,expected_policy_value

