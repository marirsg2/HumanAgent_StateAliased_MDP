"""
"""

import sys
import math
import time
import numpy as np
import random
import copy
import torch
from src.MDP_BNB import *
from src.SAMDP_policy_iteration_STANDARD_value_computation import policy_improvement as pf
from src.policy_iteration import policy_improvement as true_pf
from src.Combined_loss.minmax_normalization import *
# from src.Combined_loss.minmax_normalization import get_minmax_policy_values
# from src.Combined_loss.minmax_normalization import get_minmax_complexity_values
import os
import argparse
from domains.gridworld_domain.gridworld import GridWorldENV
from domains.amazon_domain.amazon_env_wrapper import AmazonDomainWorld
from domains.colorworld_domain.colorworld import ColorWorldENV
from domains.generic_domain.generic_env_wrapper import GenericDomainWorld


# TODO SET ENVNAME HERE. Change between gridworld, and genericworld(warehouse worker)
ENVNAME = "gridworld"
# ENVNAME = "genericworld"
#-------------------------------------

if not os.path.exists("./Results"):
    os.mkdir("./Results")


parser = argparse.ArgumentParser(description='Policy Minimization')
parser.add_argument('--env', type=str, default="gridworld", help="Env name can be g for gridworld or a for amazon")
parser.add_argument('-read', action="store_true", default="False", help="Read existing csv files for domains")
parser.add_argument('--rr', type=float, default=0, help="Random Reward Range")
args = parser.parse_args()

if ENVNAME == "gridworld" :
    env = GridWorldENV(RANDOM_NOISE_RANGE = args.rr)
elif ENVNAME == "amazon":
    env = AmazonDomainWorld(default_env=True, use_existing=args.read)
elif ENVNAME == "colorworld":
    env = ColorWorldENV()
else:
    # env = GridWorldENV()
    env = GenericDomainWorld(use_existing=True, RANDOM_NOISE_RANGE=args.rr)

NUM_STATES = env.NUM_STATES
NUM_ACTIONS = env.NUM_ACTIONS



timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = "policy_"+str(PROB_OF_RANDOM_ACTION)+"_"+str(DISCOUNT_FACTOR)+"_"+str(args.rr)+".txt"
result_file_name = "./Results/" + file_name
stdout_file_name = "./Results/Raw_output/" + file_name

# result_file_name = "./Results/policy_simplif_results_"+ENVNAME + timestr + "_NUM_STATES_" + str(NUM_STATES) +".txt"
# stdout_file_name = "./Results/Raw_output/policy_simplif_RAW_results_"+ENVNAME + timestr + "_NUM_STATES_" + str(NUM_STATES) +".txt"

# ==========================
chosen_random_seed = RANDOM_SEED
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


print (PROB_OF_RANDOM_ACTION, DISCOUNT_FACTOR, args.rr)


print("PLEASE NOTE, all console output (stdout) is being printed to the following file")
print(stdout_file_name)
sys.stdout = open(stdout_file_name,"w")

print("__file__ = ",__file__)
print("PARAMETERS")
print("REWARD_NOISE_RANGE = ", str(args.rr))
print("NUM_STATES_PER_ROW", NUM_STATES_PER_ROW)
print("WEIGHT_FOR_TRUE_POLICY_VALUE", WEIGHT_FOR_TRUE_POLICY_VALUE)
print("INCLUDE_TRUE_POLICY_IN_SEARCH", INCLUDE_TRUE_POLICY_IN_SEARCH)
print("INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH", INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH)
print("ENABLE_1HOT_POLICY_REGULARIZATION", ENABLE_1HOT_POLICY_REGULARIZATION)
# ----------------------------------
with open(result_file_name, "w") as dest:
    dest.write("PARAMETERS\n")
    dest.write("REWARD_NOISE_RANGE = " + str(args.rr) + "\n")
    dest.write("NUM_STATES_PER_ROW = " + str(NUM_STATES_PER_ROW) + "\n")
    dest.write("RANDOM_SEED = " + str(RANDOM_SEED) + "\n")
    dest.write("WEIGHT_FOR_TRUE_POLICY_VALUE = " + str(WEIGHT_FOR_TRUE_POLICY_VALUE) + "\n")
    dest.write("COMPLEXITY_INFLUENCER_SCALER_RANGE = " + str(COMPLEXITY_INFLUENCER_SCALER_RANGE) + "\n")
    dest.write("INCLUDE_TRUE_POLICY_IN_SEARCH = " + str(INCLUDE_TRUE_POLICY_IN_SEARCH) + "\n")
    dest.write("INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH = " + str(INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH) + "\n")
    dest.write("ENABLE_1HOT_POLICY_REGULARIZATION = " + str(ENABLE_1HOT_POLICY_REGULARIZATION) + "\n")
    dest.write("DISCOUNT_FACTOR = " + str(DISCOUNT_FACTOR) + "\n")
    dest.write("PROB_OF_RANDOM_ACTION = " + str(PROB_OF_RANDOM_ACTION) + "\n")
    dest.write("===========================================================\n")
#====================================================
optimal_policy,value_vector_optimal_policy = true_pf(env,RANDOM_SEED)
max_value_possible = np.mean(value_vector_optimal_policy)
start_time_for_all_settings = time.time()
best_expected_value = -math.inf
#===========================================================================
# classification_matrix
confusion_potentials_matrix = env.get_confusion_potentials_matrix()
confusion_normaliz_denom = torch.sum(confusion_potentials_matrix, dim=1)
confusion_normaliz_denom = confusion_normaliz_denom.view(-1, 1).repeat(1, NUM_STATES)
classification_matrix = confusion_potentials_matrix / confusion_normaliz_denom

# alt_guess_3d_matrix = copy.deepcopy(classification_matrix.cpu().detach().numpy())
temp_classification_matrix = copy.deepcopy(classification_matrix.cpu().detach().numpy())
alt_guess_3d_matrix = env.get_s_s_s_confusion(temp_classification_matrix)


SAS2_3d_matx = env.compute_3d_sas2_xition_matrix()

# 2d reward matrix.
rsa_reward_matrix = env.get_reward_vector(invalid_action_penalty=INVALID_ACTION_PENALTY)

max_reward_abs = torch.abs(torch.max(rsa_reward_matrix))

prob_state = torch.ones(NUM_STATES) / NUM_STATES
noop_reward_cost = NOOP_COST
noop_reward_cost_vector = torch.ones(NUM_STATES) * noop_reward_cost

action_mask_invalid_actions = torch.ones((NUM_STATES, NUM_ACTIONS))
if REMOVE_INVALID_ACTIONS:
    action_mask_invalid_actions = env.get_mask_for_invalid_actions()

all_domain_info = (env.NUM_STATES,env.NUM_ACTIONS,env.NUM_STATES_PER_ROW,classification_matrix,alt_guess_3d_matrix,SAS2_3d_matx,rsa_reward_matrix,prob_state,noop_reward_cost_vector,action_mask_invalid_actions)



# classification_matrix
confusion_potentials_matrix = env.get_confusion_potentials_matrix()
confusion_normaliz_denom = torch.sum(confusion_potentials_matrix, dim=1)
confusion_normaliz_denom = confusion_normaliz_denom.view(-1, 1).repeat(1, NUM_STATES)
classification_matrix = confusion_potentials_matrix / confusion_normaliz_denom

# alt_guess_3d_matrix = copy.deepcopy(classification_matrix.cpu().detach().numpy())
temp_classification_matrix = copy.deepcopy(classification_matrix.cpu().detach().numpy())
alt_guess_3d_matrix = env.get_s_s_s_confusion(temp_classification_matrix)


SAS2_3d_matx = env.compute_3d_sas2_xition_matrix()

# 2d reward matrix.
rsa_reward_matrix = env.get_reward_vector(invalid_action_penalty=INVALID_ACTION_PENALTY)

max_reward_abs = torch.abs(torch.max(rsa_reward_matrix))

prob_state = torch.ones(NUM_STATES) / NUM_STATES
noop_reward_cost = NOOP_COST
noop_reward_cost_vector = torch.ones(NUM_STATES) * noop_reward_cost

action_mask_invalid_actions = torch.ones((NUM_STATES, NUM_ACTIONS))
if REMOVE_INVALID_ACTIONS:
    action_mask_invalid_actions = env.get_mask_for_invalid_actions()

all_domain_info = (env.NUM_STATES,env.NUM_ACTIONS,env.NUM_STATES_PER_ROW,classification_matrix,alt_guess_3d_matrix,SAS2_3d_matx,rsa_reward_matrix,prob_state,noop_reward_cost_vector,action_mask_invalid_actions)



optimal_policy,value_vector_optimal_policy = true_pf(env,RANDOM_SEED)
max_value_possible = np.mean(value_vector_optimal_policy)
start_time_for_all_settings = time.time()
best_expected_value = -math.inf



#===========================================================================
for new_discount in [DISCOUNT_FACTOR]: #tried using global and iterating, but other parts of the code pull from config anew, and this takes the old value. NEED to pass new value to all functions. Needs checking
    # DISCOUNT_FACTOR = new_discount DOES NOT WORK, manually change. Global does not work
    for new_L1_conf_potentials_divisor in L1_CONF_POTENTIALS_DIVISOR_RANGE:
        env.set_potentials_divisor(new_L1_conf_potentials_divisor)
        complexity_weight = 0 #old code
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("complexity_weight", complexity_weight)
        # print("DISCOUNT_FACTOR", DISCOUNT_FACTOR)
        # print("L1_CONF_POTENTIALS_DIVISOR", new_L1_conf_potentials_divisor)
        print("no-op reward(cost)", NOOP_COST)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        with open(result_file_name, "a") as dest:
            dest.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
            dest.write("complexity_weight = " + str(complexity_weight) + "\n")
            dest.write("NOOP_COST = " + str(NOOP_COST) + "\n")
            dest.write("===========================================================\n")

        for trial_idx in range(TRIALS_PER_SETTING):
            print("trial_idx = ", trial_idx)
            with open(result_file_name, "a") as dest:
                dest.write(" trial_idx = " + str(trial_idx) + "\n")
                dest.write("===========================================================\n")
            start_time_of_policy_iter_search = time.time()
            policy_matrix,final_weighted_value,classification_matrix,alt_guess_3d_matrix,true_policy_value,expected_policy_value = \
                pf(all_domain_info,max_value_possible,complexity_weight=complexity_weight,printout_filename = result_file_name)
            print("policy matrix",policy_matrix)
            print("Time taken (seconds) = ", time.time() - start_time_of_policy_iter_search)

            # ------------PRINTOUTS------------
            print("true_policy_matrix")
            print(policy_matrix)
            env.translate_and_print_policy(policy_matrix)
            print("===============")
            print("weighted value = ", final_weighted_value)
            print("1-hot policy, true_value = ", true_policy_value)
            print("1-hot policy, expected_value = ", expected_policy_value)
            if expected_policy_value > best_expected_value:
                best_expected_value = expected_policy_value
            print("TRUE CONFUSION SCORE = ",
                  compute_true_policy_confusion_score(classification_matrix,alt_guess_3d_matrix, policy_matrix))
            print("Time taken (seconds) = ", time.time() - start_time_of_policy_iter_search)
            print("==========")
            with open(result_file_name, "a") as dest:
                dest.write("NOTE THE INVALID ACTION PENALTY IS REMOVED for the value computation" + "\n")
                dest.write("weighted value = " + str(final_weighted_value.item()) + "\n")
                dest.write("1-hot policy true_value = " + str(true_policy_value.item()) + "\n")
                dest.write("1-hot policy expected_value = " + str(expected_policy_value.item()) + "\n")
                try:
                    dest.write("TRUE CONFUSION SCORE = " + str(
                        compute_true_policy_confusion_score(classification_matrix,alt_guess_3d_matrix,
                                                            policy_matrix).item()) + "\n")
                except:
                    dest.write("TRUE CONFUSION SCORE = " + str(
                        compute_true_policy_confusion_score(classification_matrix,alt_guess_3d_matrix, policy_matrix)) + "\n")
                dest.write("Time taken (seconds) = " + str(time.time() - start_time_of_policy_iter_search) + "\n")
                dest.write("===========================================================\n")
        #=====HILL CLIMBING IS OVER
        print(" The best value from hill climbing was = ", best_expected_value)
        #========================================================================
        print("-------------BRANCH AND BOUND ---------------------------------")


        # todo test calling BNB here
        noop_action_transition_effect = np.zeros((rsa_reward_matrix.shape[0], rsa_reward_matrix.shape[0]))
        np.fill_diagonal(noop_action_transition_effect, 1)
        # add no-op costs to rsa too
        rsa_w_noop_cost = np.append(rsa_reward_matrix, np.transpose(np.atleast_2d(noop_reward_cost_vector), (1, 0)), axis=1)

        # print("---REMOVE THIS CODE----")
        # prob_state = prob_state * 0
        # prob_state[0] = 1  # TODO NOTE WE PUT ALL THE WEIGHT on the first state
        #=========================

        prob_state2 = prob_state.cpu().detach().numpy()
        prob_state2 = prob_state2 / prob_state2.sum()
        prob_state2 = prob_state2 / prob_state2.sum()
        classification_matrix2 = np.array(classification_matrix.cpu().detach().numpy(), dtype=np.float64)
        classification_matrix2 = classification_matrix2 / classification_matrix2.sum(axis=1)[:, np.newaxis]
        classification_matrix2 = classification_matrix2 / classification_matrix2.sum(axis=1)[:,
                                                          np.newaxis]  # repeating seems to ensure that none is above 0.
        # TODO ADD CHECK for any greater than 1 in the sum. somehow torch to numpy conversion results in the probabilities no longer adding to 1 !!
        SAS2_3d_matx2 = np.array(SAS2_3d_matx.cpu().detach().numpy(), dtype=np.float64)
        SAS2_3d_matx2 = SAS2_3d_matx2 / SAS2_3d_matx2.sum(axis=2)[:, :, np.newaxis]
        SAS2_3d_matx2 = SAS2_3d_matx2 / SAS2_3d_matx2.sum(axis=2)[:, :, np.newaxis]
        alt_guess_3d_matrix2 = alt_guess_3d_matrix / alt_guess_3d_matrix.sum(axis=2)[:, :, np.newaxis]
        alt_guess_3d_matrix2 = alt_guess_3d_matrix2 / alt_guess_3d_matrix2.sum(axis=2)[:, :, np.newaxis]
        #The above lines were necessary because somehow torch to numpy conversion results in the probabilities no longer adding to 1 !!

        noop_reward_cost_vector2 = np.array(noop_reward_cost_vector.cpu().detach().numpy(), dtype=np.float64)


        state_order_idx_list = list(reversed(range(prob_state2.shape[0])))
        state_score_vector = compute_state_scores_for_bnb(classification_matrix2,rsa_w_noop_cost,prob_state2)
        list_state_idx_score_pair = [(x,state_score_vector[x]) for x in range(state_score_vector.shape[0])]
        state_order_idx_list = [x[0] for x in sorted(list_state_idx_score_pair,key=lambda x:x[1],reverse=True)]

        hasa_problem = HASA_MDP(prob_state2, SAS2_3d_matx2, noop_action_transition_effect, noop_reward_cost_vector2,
                                rsa_w_noop_cost, DISCOUNT_FACTOR, classification_matrix2, alt_guess_3d_matrix2,
                                np.ones(action_mask_invalid_actions.shape),
                                np.zeros(rsa_reward_matrix.shape),
                                state_order_idx_list=state_order_idx_list, null_actions_enabled=True)
        # def __init__(self, state_initial_prob, SAS2_3d_matx,noop_action_effect, noop_reward_cost_vector,
        #              rsa_reward_matrix,discount_gamma, classification_matrix, alt_guess_3d_matrix, action_mask_invalid_actions,
        #              partial_policy, state_order_idx_list = None, next_unassigned_state_idx = 0, noop_prob_scaler= 1, null_actions_enabled=True):
        import time
        start_time = time.time()
        # results = pybnb.misc.create_command_line_solver(hasa_problem)
        # results = pybnb.solve(hasa_problem, comm=None,queue_strategy="bound")  # TODO update the command line arguments to take in HC result --best-objective BEST_OBJ
        results = pybnb.solve(hasa_problem, comm=None,queue_strategy="bound",best_objective = best_expected_value-1,
                              log_filename=result_file_name.replace(".txt","_BNB.txt"))  # we subtract 0.1 from the best value to avoid issues with numerical computation
        # results = pybnb.solve(hasa_problem, comm=None,best_objective=3.0595)  # TODO update the best objective to take the HC result
        print("---BNB takes %s seconds ---" % (time.time() - start_time))
        print("results from BnB with HASA BnB controller")
        try:
            curr_policy = torch.Tensor(results.best_node.state[9])
            print(curr_policy)
            env.translate_and_print_policy(curr_policy)
            null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(curr_policy, classification_matrix,
                                                                                             alt_guess_3d_matrix,
                                                                                             noop_prob_scaler=1.0)
            expected_policy_value = compute_expected_policy_value(classification_matrix,
                                                                  rsa_reward_matrix, curr_policy, null_action_likelihood_vector,
                                                                  noop_reward_cost_vector, SAS2_3d_matx, prob_state)
        except:
            pass
        print("Expected Value with HASA BnB Controller =", expected_policy_value)



    print("time taken for all settings = ", time.time()-start_time_for_all_settings)

