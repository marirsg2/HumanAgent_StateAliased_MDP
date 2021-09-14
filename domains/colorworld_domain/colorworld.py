from src.Combined_loss.config import *
# to support legacy code.
import torch
import numpy as np


class ColorWorldENV :

    def __init__(self):
        self.NUM_ACTIONS = 4
        self.NUM_STATES = 6

    # ===========================================================================
    # third.
    def compute_3d_sas2_xition_matrix(self):
        """

        :param policy_matx:
        :return:
        """
        xition_prob_by_SxAxS2_matx = torch.ones((self.NUM_STATES, self.NUM_ACTIONS, self.NUM_STATES))
        xition_prob_by_SxAxS2_matx *= 1/self.NUM_STATES #all transitions are equally likely
        return xition_prob_by_SxAxS2_matx

    #=====================================================================
    # first
    def get_confusion_potentials_matrix(self):
        """

        :return:
        """
        confusion_potentials_matrix = torch.zeros((self.NUM_STATES, self.NUM_STATES), dtype=torch.float32)
        #-------color pair 1
        confusion_potentials_matrix[0][0] = 0.5
        confusion_potentials_matrix[0][1] = 0.5
        confusion_potentials_matrix[1][0] = 0.5
        confusion_potentials_matrix[1][1] = 0.5
        #----------------
        confusion_potentials_matrix[2][2] = 0.5
        confusion_potentials_matrix[2][3] = 0.5
        confusion_potentials_matrix[3][2] = 0.5
        confusion_potentials_matrix[3][3] = 0.5
        #--------------------------------------
        confusion_potentials_matrix[4][4] = 0.5
        confusion_potentials_matrix[4][5] = 0.5
        confusion_potentials_matrix[5][4] = 0.5
        confusion_potentials_matrix[5][5] = 0.5

        return confusion_potentials_matrix


    #=====================================================================
    # second
    def get_reward_vector(self,invalid_action_penalty = -0.1, goal_state_reward = 10):
        """
        :return:
        :summary: 2d matx SxA
        """

        #---------------
        # reward_matrix = torch.rand(NUM_STATES,NUM_ACTIONS)
        reward_matrix = -1*torch.ones(self.NUM_STATES,self.NUM_ACTIONS)
        #same encoding as in the config file for up, right, down , left
        reward_matrix[0][ACTION_RIGHT] = 1.0
        #------------
        reward_matrix[1][ACTION_UP] = 1.1
        reward_matrix[1][ACTION_RIGHT] = 1.0
        # ------------
        reward_matrix[2][ACTION_DOWN] = 1.0
        # ------------
        reward_matrix[3][ACTION_DOWN] = 1.0
        reward_matrix[3][ACTION_RIGHT] = 1.1
        # ------------
        reward_matrix[4][ACTION_UP] = 1.0
        reward_matrix[4][ACTION_LEFT] = 1.1
        # ------------
        reward_matrix[5][ACTION_UP] = 1.0
        # ------------
        return reward_matrix
    #=============================================================

    def translate_and_print_policy(self, policy_matrix, cutoff = 0.1):
        """

        :param true_policy_matrix:
        :return:
        """
        action_map = ["ACTION_UP","ACTION_RIGHT","ACTION_DOWN","ACTION_LEFT"]
        list_version_policy = list(policy_matrix)
        translated_policy_list = ["" for x in range(self.NUM_STATES)]
        for state_idx in range(len(list_version_policy)):
            row = list_version_policy[state_idx]
            temp_data = row.cpu().detach().numpy()
            translated_policy_list[state_idx] = ",".join([str((action_map[action_idx],temp_data[action_idx])) for action_idx in range(len(row))
                     if temp_data[action_idx] > cutoff])
        #---end for loop
        translated_policy_list = np.array(translated_policy_list)
        print(translated_policy_list)
    #=============================================================  

    def remove_penalty(self, rsa_reward_vector):
        """

        :param reward_vector:
        :return:
        """
        pass

    #=====================================================================

    def get_mask_for_invalid_actions(self):
        """

        :return:
        """
        mask_tensor = torch.ones([self.NUM_STATES, self.NUM_ACTIONS], dtype=torch.float32)
        #all actions are valid
        return mask_tensor
    #=============================================================

    def remove_invalid_actions(self, action_potential_array):
        """
        :param action_potential_array:
        :return:
        """
        return action_potential_array

    #=====================================================================
    def compute_transition_likelihood_for_random_action(self, source_state_idx,possible_actions):
        """

        :param state_idx:
        :param possible_actions:
        :return:
        """
        transition_state_probabilities = torch.ones(self.NUM_STATES)*1/NUM_STATES
        return transition_state_probabilities










