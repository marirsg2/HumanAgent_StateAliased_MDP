"""
"""
import numpy as np
import pybnb
import copy
from src.policy_matrix_functions import *

class HASA_MDP(pybnb.Problem):
    def __init__(self, state_initial_prob, SAS2_3d_matx,noop_action_effect, noop_reward_cost_vector,
                 rsa_reward_matrix,discount_gamma, classification_matrix, alt_guess_3d_matrix, action_mask_invalid_actions,
                 partial_policy, state_order_idx_list = None, next_unassigned_state_idx = 0, noop_prob_scaler= 1, null_actions_enabled=True):
        """
        noop_prob_scaler = 0 would make it no no-op action.
        """
        # assert W >= 0
        # assert len(v) == len(w)
        self.state_initial_prob  = state_initial_prob
        self.SAS2_3d_matx = SAS2_3d_matx
        self.noop_action_effect =noop_action_effect
        self.noop_reward_cost_vector = noop_reward_cost_vector
        self.rsa_reward_matrix = rsa_reward_matrix
        self.discount_gamma = discount_gamma
        self.confusion_classification_matrix = classification_matrix
        self.alt_guess_3d_matrix = alt_guess_3d_matrix
        self.action_mask_invalid_actions = action_mask_invalid_actions
        self.current_policy = partial_policy
        self.null_actions_enabled = null_actions_enabled
        self.noop_prob_scaler = noop_prob_scaler
        self.state_order_idx_list = state_order_idx_list
        if state_order_idx_list == None:
            self.state_order_idx_list = list(range(state_initial_prob.shape[0]))
        self.next_unassigned_state_idx = next_unassigned_state_idx
        #todo GET INTERMEDIATE OBJECTIVE WORKING !! very helpful for when using only BnB to get some best value and terminate anytime

    def sense(self):
        return pybnb.maximize


    #=======================
    def objective(self):
        """


        """
        if np.all(np.sum(self.current_policy,axis=1) - np.ones(self.current_policy.shape[0]) == 0): #then it is a complete policy





            null_action_likelihood_vector = numpy_compute_null_action_likelihoods_for_states_wDelayScaler(self.current_policy,
                                                                                                          self.confusion_classification_matrix,
                                                                                                          self.alt_guess_3d_matrix,
                                                                                                          self.noop_prob_scaler)
            only_policy_actions_rsa = self.rsa_reward_matrix[:,:-1] #remove the delay action
            return numpy_compute_expected_policy_value(self.confusion_classification_matrix,
                                                 only_policy_actions_rsa, self.current_policy,
                                                 null_action_likelihood_vector, self.noop_reward_cost_vector, self.SAS2_3d_matx, self.state_initial_prob)


        return self.infeasible_objective() #todo improve upon this. Use a random completion of the policy, save it, incase it is the best :-), and compute the policy value as use as objective.
    #=============================================


    #=============================================

    def bound(self,iteration_cutoff = 1000):
        """
         When we get the values from the value iteration code, we get both an upper and lower bound on what the
         value could be. If the lowest is greater than the current lower bound, we stop and return that as the value.
         If the upper is lower than the current lower bound, we stop as well. Else we continue till the chosen epsilon
         and return the upperbound. If that is greater than the lower bound, we continue the branch
        """
        # translate the partial policy to updated SAS2 matrix
        if np.all(np.sum(self.current_policy, axis=1) - np.ones(self.current_policy.shape[0]) == 0):
            # print("Catch")
            return self.objective()

        # print("remove code !")
        # self.current_policy[:,0] = 1

        original_rsa = self.rsa_reward_matrix[:,:-1]
        modified_rsa, modified_action_sas_w_delay = compute_updated_RSA_and_SAS_with_delay(self.current_policy, original_rsa,self.SAS2_3d_matx, self.confusion_classification_matrix,
                                                                             self.alt_guess_3d_matrix, self.noop_reward_cost_vector,self.noop_action_effect,
                                                                             noop_prob_scaler=1.0)
        assert np.all(np.sum(modified_action_sas_w_delay,axis=2)-COMPUTATIONAL_TOLERANCE<1)
        assert np.all(np.sum(modified_action_sas_w_delay,axis=2)+COMPUTATIONAL_TOLERANCE>1)
        assert np.all(modified_action_sas_w_delay-COMPUTATIONAL_TOLERANCE <1)
        assert np.all(modified_action_sas_w_delay+COMPUTATIONAL_TOLERANCE > 0)

        #TODO UPDATE THE REWARDS TOO !! to reflect the new transitions
        state_set = list(range(self.current_policy.shape[0]))
        action_set = list(range(self.current_policy.shape[1] + 1))  # +1 for the delay action
        result_dict = ValueIteration(state_set, action_set, modified_rsa, modified_action_sas_w_delay, epsilon=0.001,
                                gamma=self.discount_gamma, ITERATION_CUTOFF=iteration_cutoff, d_start=None)
        bound_value = np.sum( result_dict["maxV"]*self.state_initial_prob)
        return bound_value


    def save_state(self, node):
        node.state = (
                self.state_initial_prob ,
                self.SAS2_3d_matx ,
                self.noop_action_effect ,
                self.noop_reward_cost_vector ,
                self.rsa_reward_matrix ,
                self.discount_gamma,
                self.confusion_classification_matrix ,
                self.alt_guess_3d_matrix ,
                self.action_mask_invalid_actions ,
                self.current_policy ,
                self.null_actions_enabled ,
                self.noop_prob_scaler,
                self.state_order_idx_list,
                self.next_unassigned_state_idx,
                 )


    def load_state(self, node):
        (
            self.state_initial_prob,
            self.SAS2_3d_matx,
            self.noop_action_effect,
            self.noop_reward_cost_vector,
            self.rsa_reward_matrix,
            self.discount_gamma,
            self.confusion_classification_matrix,
            self.alt_guess_3d_matrix,
            self.action_mask_invalid_actions,
            self.current_policy,
            self.null_actions_enabled,
            self.noop_prob_scaler,
            self.state_order_idx_list,
            self.next_unassigned_state_idx,
            ) = node.state


    def branch(self):
        """
         This should yield the possible actions for the next state.
        """
        if self.next_unassigned_state_idx == len(self.state_order_idx_list):
            return
        for action_idx in range(self.current_policy.shape[1]):
            child = pybnb.Node()
            new_policy = copy.deepcopy(self.current_policy)
            new_policy[self.state_order_idx_list[self.next_unassigned_state_idx]][action_idx] = 1
            # print("===========================")
            # print(new_policy)
            # print("===========================")
            child.state = (
                self.state_initial_prob ,
                self.SAS2_3d_matx ,
                self.noop_action_effect ,
                self.noop_reward_cost_vector ,
                self.rsa_reward_matrix ,
                self.discount_gamma,
                self.confusion_classification_matrix ,
                self.alt_guess_3d_matrix ,
                self.action_mask_invalid_actions ,
                new_policy ,
                self.null_actions_enabled ,
                self.noop_prob_scaler,
                self.state_order_idx_list,
                self.next_unassigned_state_idx +1, #+1 for the next node
                 )
            #update the policy action

            yield child



if __name__ == "__main__":
    import pybnb.misc




    results = pybnb.solve(problem, comm=None, best_objective = 2621)
    print(results)