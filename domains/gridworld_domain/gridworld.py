from src.Combined_loss.config import * 
# to support legacy code.

import time
import torch
import numpy as np
import math
import random
from torch.autograd import Variable
from torch import autograd
from src.Combined_loss.minmax_normalization import *
import sys


class GridWorldENV : 

    def __init__(self, RANDOM_NOISE_RANGE = 0):
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_STATES = NUM_STATES
        self.NUM_STATES_PER_ROW = NUM_STATES_PER_ROW
        self.L1_divisor = 1
        self.RANDOM_NOISE_RANGE = RANDOM_NOISE_RANGE

    # ===========================================================================
    # third.
    def compute_3d_sas2_xition_matrix(self):
        """

        :param policy_matx:
        :return:
        """
        xition_prob_by_SxAxS2_matx = torch.zeros((self.NUM_STATES, self.NUM_ACTIONS, self.NUM_STATES)) # will be filled
        # now fix the policy such that each action transition has 1.0 - x odds of transitioning in the right direction.
        for source_state_idx in range(self.NUM_STATES):
            valid_action_set = set()
            invalid_action_set = set()
            # if the action is up, then we go up by STATES PER ROW. So from 4 (row 1, col 0) moving up is to state 0, <row0,col0>
            for action_idx in range(self.NUM_ACTIONS):
                # we have already chosen the action !!
                # just check if invalid and handle, i.e. if the invalid one is the current action
                valid_action = True
                # -----------------------------
                if int(source_state_idx / self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_UP:
                    valid_action = False
                elif int(source_state_idx / self.NUM_STATES_PER_ROW) == self.NUM_STATES_PER_ROW - 1 and action_idx == ACTION_DOWN:
                    valid_action = False
                # -----------------------------
                if int(source_state_idx % self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_LEFT:
                    valid_action = False
                elif int(source_state_idx % self.NUM_STATES_PER_ROW) == self.NUM_STATES_PER_ROW - 1 and action_idx == ACTION_RIGHT:
                    valid_action = False
                # end elif
                # we will need the random transition likelihoods even if valid action as there is always a PROB_OF_WRONG_XITION
                if valid_action:
                    valid_action_set.add(action_idx)
                else:
                    invalid_action_set.add(action_idx)
                # end else
            # end for loop through all actions of the domain (not just state)
            # ---now we have valid and invalid actions.
            for action_idx in invalid_action_set:
                #if invalid action stay in the same state
                xition_prob_by_SxAxS2_matx[source_state_idx][action_idx][source_state_idx] = 1.0


            for action_idx in valid_action_set:
                # it is a valid action and we compute the state probabilities accordingly
                transition_state_probabilities = torch.zeros(self.NUM_STATES)
                next_state = self.get_deterministic_next_state_with_move(source_state_idx, action_idx)
                transition_state_probabilities[next_state] = 1 - PROB_OF_RANDOM_ACTION
                for second_action_idx in valid_action_set:
                    other_possible_next_state = self.get_deterministic_next_state_with_move(source_state_idx,
                                                                                       second_action_idx)
                    transition_state_probabilities[other_possible_next_state] += PROB_OF_RANDOM_ACTION / len(
                        valid_action_set)
                # ---end for loop
                xition_prob_by_SxAxS2_matx[source_state_idx][action_idx] = transition_state_probabilities
            # ---end for through valid actions
        # ---end for source_state_idx in range(NUM_STATES):
        # now we have finished computing xition_prob_by_SxAxS2_matx

        #Todo note, this makes the very last state an absorbing state
        xition_prob_by_SxAxS2_matx[-1,:,:] = 0
        xition_prob_by_SxAxS2_matx[-1,:,-1] = 1

        return xition_prob_by_SxAxS2_matx

    #=====================================================================
    # first
    def get_confusion_potentials_matrix(self):
        """

        :return:
        """

        torch.zeros((self.NUM_STATES, self.NUM_STATES), dtype=torch.float32)
        confusion_potentials_matrix = torch.zeros((self.NUM_STATES, self.NUM_STATES), dtype=torch.float32)
        L1_distance_matrix = torch.zeros((self.NUM_STATES, self.NUM_STATES), dtype=torch.float32)
        all_state_idx_list = set(range(self.NUM_STATES))
        for first_state_idx in all_state_idx_list:
            first_row_idx = int(first_state_idx/self.NUM_STATES_PER_ROW)
            first_col_idx = int(first_state_idx % self.NUM_STATES_PER_ROW)
            valid_state_idx_list = all_state_idx_list
            # for  confusion with just the plus (+) shape neighborhood tiles
            # valid_state_idx_list = [x for x in \
            #             [first_state_idx,first_state_idx+1,first_state_idx-1,first_state_idx + NUM_STATES_PER_ROW,first_state_idx-NUM_STATES_PER_ROW]
            #                         if x >=0 and x <= NUM_STATES-1 ]
            #===========================
            for second_state_idx in valid_state_idx_list:

                second_row_idx = int(second_state_idx / self.NUM_STATES_PER_ROW)
                second_col_idx = int(second_state_idx % self.NUM_STATES_PER_ROW)
                L1_distance_matrix[first_state_idx][second_state_idx] = abs(first_row_idx-second_row_idx) + \
                                                            abs(first_col_idx - second_col_idx) + 1 #+1 to avoid div by zero
                # L1_distance_matrix[first_state_idx][second_state_idx] = torch.pow(L1_distance_matrix[first_state_idx][second_state_idx],self.L1_divisor) #else it's too tightly coupled
                L1_distance_matrix[first_state_idx][second_state_idx] = torch.pow(L1_distance_matrix[first_state_idx][second_state_idx],5) #else it's too tightly coupled
            # --end for
        # end for


        #similarity of two states is defined by their distance to each other, and with respect to other states
        # sim_mask = torch.where(L1_distance_matrix == 0,0,1)
        # dummy_plus_1 = torch.where(L1_distance_matrix == 0, 1, 0)
        # confusion_potentials_matrix = torch.pow(L1_distance_matrix+dummy_plus_1,-1)
        confusion_potentials_matrix = torch.pow(L1_distance_matrix,-1)
        # confusion_potentials_matrix *= sim_mask
        return confusion_potentials_matrix

        # So self-similarity potential is 1, then one step away have similarity potential 1/2, two steps is 1/3 etc.

        #=============================
        # all_state_idx_list = set(range(NUM_STATES))
        # while len(all_state_idx_list) > 0:
        #     num_choose = random.randint(1, len(all_state_idx_list))
        #     chosen_states = set(random.sample(all_state_idx_list, num_choose))
        #     all_state_idx_list.difference_update(chosen_states)
        #     # assign confusion potential for this set of states
        #     for single_state_idx in chosen_states:
        #         for second_state_idx in chosen_states:
        #             # --
        #             confusion_potentials_matrix[single_state_idx][second_state_idx] += random.random()
        #         # --end for
        #     # end for
        # # end while

        # ---------------------------
        # completely random confusion matrix - extreme scenario
        # confusion_potentials_matrix = torch.rand((NUM_STATES, NUM_STATES), dtype=torch.float32)
        # ---------------------------
        # for all states in a column to be confusing
        # confusion_potentials_matrix = torch.zeros((NUM_STATES, NUM_STATES), dtype=torch.float32)
        # for i in range(NUM_STATES_PER_ROW):
        #     for j in range(NUM_STATES_PER_ROW):
        #         row_idx = i + j * NUM_STATES_PER_ROW
        #         confusion_potentials_matrix[row_idx][i::NUM_STATES_PER_ROW] = 1
        # # we add the identity matrix so the state would always be self-aliased the most with itself. Not necessary
        # confusion_potentials_matrix += torch.eye(NUM_STATES, dtype=torch.float32)




    #=====================================================================
    def set_potentials_divisor(self,divisor):
        self.L1_divisor = divisor
    #=====================================================================
    # second
    def get_reward_vector(self,invalid_action_penalty = -0.0, goal_state_reward = 100):
        """
        :return:
        :summary: 2d matx SxA
        """
        random_noise_range = self.RANDOM_NOISE_RANGE
        # random_noise_range = 10

        torch.manual_seed(RANDOM_SEED)

        #---------------
        reward_matrix = torch.rand(NUM_STATES,NUM_ACTIONS)*random_noise_range - random_noise_range/2
        # reward_matrix = torch.zeros(self.NUM_STATES,self.NUM_ACTIONS)
        # reward_matrix[NUM_STATES-1] += goal_state_reward
        # reward_matrix[NUM_STATES-NUM_STATES_PER_ROW] += goal_state_reward

        #--------goal config 1
        #reward when we transition into the corner state, not from the corner state. less intuitive
        reward_matrix[NUM_STATES - 2][ACTION_RIGHT] += goal_state_reward
        reward_matrix[NUM_STATES - NUM_STATES_PER_ROW-1][ACTION_DOWN] += goal_state_reward
        #TODO also make them absorbing states, and remove the other as the absorbing state


        #--------goal config 2
        #goal in opposite diagonals
        # reward for bottom left
        # reward_matrix[NUM_STATES - 2*NUM_STATES_PER_ROW][ACTION_DOWN] += goal_state_reward
        # reward_matrix[NUM_STATES - NUM_STATES_PER_ROW+1][ACTION_LEFT] += goal_state_reward
        #--reward for top right
        # reward_matrix[NUM_STATES_PER_ROW - 1][ACTION_RIGHT] += goal_state_reward
        # reward_matrix[2*NUM_STATES_PER_ROW -1][ACTION_UP] += goal_state_reward

        #TODO also make them absorbing states, and remove the other as the absorbing state



        #---------landmark goals
        #this is to add one more rewards in between, helps converge faster
        # reward_matrix[NUM_STATES_PER_ROW-int(NUM_STATES_PER_ROW/2)][ACTION_RIGHT] += goal_state_reward/4
        # reward_matrix[2*NUM_STATES_PER_ROW][ACTION_DOWN] += goal_state_reward/4

        for source_state_idx in range(self.NUM_STATES):
            valid_action_set = set()
            invalid_action_set = set()
            # if the action is up, then we go up by STATES PER ROW. So from 4 (row 1, col 0) moving up is to state 0, <row0,col0>
            for action_idx in range(self.NUM_ACTIONS):
                # we have already chosen the action !!
                # just check if invalid and handle, i.e. if the invalid one is the current action
                valid_action = True
                # -----------------------------
                if int(source_state_idx / self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_UP:
                    valid_action = False
                elif int(source_state_idx / self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_DOWN:
                    valid_action = False
                # -----------------------------
                if int(source_state_idx % self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_LEFT:
                    valid_action = False
                elif int(source_state_idx % self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_RIGHT:
                    valid_action = False
                # end elif
                # we will need the random transition likelihoods even if valid action as there is always a PROB_OF_WRONG_XITION
                if valid_action:
                    valid_action_set.add(action_idx)
                else:
                    invalid_action_set.add(action_idx)
                #end else
            #end for loop through all actions of the domain (not just state)
            # ---now we have valid and invalid actions.
            for action_idx in invalid_action_set:
                reward_matrix[source_state_idx][action_idx] = invalid_action_penalty
            #end for loop
        #end of for loop through states

        return reward_matrix


    def print_numpy_policy(self, policy) : 
        action_map = ["ACTION_UP","ACTION_RIGHT","ACTION_DOWN","ACTION_LEFT"]

        pp = [["" for x in range(self.NUM_STATES_PER_ROW)] for y in range(self.NUM_STATES_PER_ROW)]

        for idx, ps in enumerate(policy) : 

                action = np.argmax(ps)
                print (ps, action, action_map[action])
                row_idx = int(idx/self.NUM_STATES_PER_ROW)
                col_idx = int(idx % self.NUM_STATES_PER_ROW)

                pp[row_idx][col_idx] = action_map[action]

        return pp 


    def translate_and_print_policy(self, policy_matrix, cutoff = 0.1):
        """

        :param true_policy_matrix:
        :return:
        """
        action_map = ["ACTION_UP","ACTION_RIGHT","ACTION_DOWN","ACTION_LEFT"]
        list_version_policy = list(policy_matrix)
        translated_policy_matrix = [["" for x in range(self.NUM_STATES_PER_ROW)] for y in range(self.NUM_STATES_PER_ROW)]
        for state_idx in range(len(list_version_policy)):
            row = list_version_policy[state_idx]
            row_idx = int(state_idx/self.NUM_STATES_PER_ROW)
            col_idx = int(state_idx % self.NUM_STATES_PER_ROW)
            try : 
                temp_data = row.cpu().detach().numpy()
            except : 
                temp_data = row

            translated_policy_matrix[row_idx][col_idx] = ",".join([str((action_map[action_idx],temp_data[action_idx])) for action_idx in range(len(row))
                     if temp_data[action_idx] > cutoff])
        #---end for loop
        translated_policy_matrix = np.array(translated_policy_matrix)
        print(translated_policy_matrix)

    def remove_penalty(self, rsa_reward_vector):
        """

        :param reward_vector:
        :return:
        """
        for source_state_idx in range(self.NUM_STATES):
            valid_action_set = set()
            invalid_action_set = set()
            # if the action is up, then we go up by STATES PER ROW. So from 4 (row 1, col 0) moving up is to state 0, <row0,col0>
            for action_idx in range(self.NUM_ACTIONS):
                # we have already chosen the action !!
                # just check if invalid and handle, i.e. if the invalid one is the current action
                valid_action = True
                # -----------------------------
                if int(source_state_idx / self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_UP:
                    valid_action = False
                elif int(source_state_idx / self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_DOWN:
                    valid_action = False
                # -----------------------------
                if int(source_state_idx % self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_LEFT:
                    valid_action = False
                elif int(source_state_idx % self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_RIGHT:
                    valid_action = False
                # end elif
                # we will need the random transition likelihoods even if valid action as there is always a PROB_OF_WRONG_XITION
                if valid_action:
                    valid_action_set.add(action_idx)
                else:
                    invalid_action_set.add(action_idx)
                #end else
            #end for loop through all actions of the domain (not just state)
            # ---now we have valid and invalid actions.
            for action_idx in invalid_action_set:
                rsa_reward_vector[source_state_idx][action_idx] = 0
            #end for loop
        #end of for loop through states

    #=====================================================================
    #=====================================================================
    #=====================================================================
    # HELPER FUNCTIONS 
    #=====================================================================
    #=====================================================================
    #=====================================================================
    def get_deterministic_next_state_with_move(self, state_idx, action_idx):
        """

        :param state_idx:
        :param action_idx:
        :return: a vector of probabilities indicating the likely next states
        :summary:
        note we assume the num states is def by var NUM_STATES
        """
        row_idx_offset = int(state_idx/self.NUM_STATES_PER_ROW) # when moving up or down
        col_idx_offset = int(state_idx%self.NUM_STATES_PER_ROW) # when moving left or right
        if action_idx == ACTION_UP:
            row_idx_offset -=  1
        elif action_idx == ACTION_DOWN:
            row_idx_offset +=  1
        elif action_idx == ACTION_LEFT:
            col_idx_offset -= 1
        elif action_idx == ACTION_RIGHT:
            col_idx_offset += 1
        else:
            raise Exception("Unknown action index")

        if row_idx_offset < 0 or row_idx_offset >= self.NUM_STATES_PER_ROW: #IF EQUAL TO NUM_STATES_PER_ROW then error
            raise Exception("Action effect led to invalid (negative) state index")
        if col_idx_offset < 0 or col_idx_offset >= self.NUM_STATES_PER_ROW: #IF EQUAL TO NUM_STATES_PER_ROW then error
            raise Exception("Action effect led to invalid (negative) state index")

        resultant_state_idx = int(row_idx_offset*self.NUM_STATES_PER_ROW + col_idx_offset)
        return resultant_state_idx




    def get_mask_for_invalid_actions(self):
        """

        :return:
        """
        mask_tensor = torch.ones([self.NUM_STATES, self.NUM_ACTIONS], dtype=torch.float32)
        for source_state_idx in range(self.NUM_STATES):
            valid_action_set = set()
            invalid_action_set = set()
            # if the action is up, then we go up by STATES PER ROW. So from 4 (row 1, col 0) moving up is to state 0, <row0,col0>
            for action_idx in range(self.NUM_ACTIONS):
                # we have already chosen the action !!
                # just check if invalid and handle, i.e. if the invalid one is the current action
                valid_action = True
                # -----------------------------
                if int(source_state_idx / self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_UP:
                    valid_action = False
                elif int(source_state_idx / self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_DOWN:
                    valid_action = False
                # -----------------------------
                if int(source_state_idx % self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_LEFT:
                    valid_action = False
                elif int(source_state_idx % self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_RIGHT:
                    valid_action = False
                # end elif
                # we will need the random transition likelihoods even if valid action as there is always a PROB_OF_WRONG_XITION
                if valid_action:
                    valid_action_set.add(action_idx)
                else:
                    invalid_action_set.add(action_idx)
                # end else
            # end for loop through all actions of the domain (not just state)
            # ---now we have valid and invalid actions.
            for action_idx in invalid_action_set:
                mask_tensor[source_state_idx][action_idx] = 0.0
            # --end for
        #end outer for
        return mask_tensor


    def remove_invalid_actions(self, action_potential_array):
        """
        :param action_potential_array:
        :return:
        """
        mask_tensor = torch.ones([self.NUM_STATES,self.NUM_ACTIONS],dtype=torch.float32)
        for source_state_idx in range(NUM_STATES):
            valid_action_set = set()
            invalid_action_set = set()
            # if the action is up, then we go up by STATES PER ROW. So from 4 (row 1, col 0) moving up is to state 0, <row0,col0>
            for action_idx in range(self.NUM_ACTIONS):
                # we have already chosen the action !!
                # just check if invalid and handle, i.e. if the invalid one is the current action
                valid_action = True
                # -----------------------------
                if int(source_state_idx / self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_UP:
                    valid_action = False
                elif int(source_state_idx / self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_DOWN:
                    valid_action = False
                # -----------------------------
                if int(source_state_idx % self.NUM_STATES_PER_ROW) == 0 and action_idx == ACTION_LEFT:
                    valid_action = False
                elif int(source_state_idx % self.NUM_STATES_PER_ROW) == NUM_STATES_PER_ROW - 1 and action_idx == ACTION_RIGHT:
                    valid_action = False
                # end elif
                # we will need the random transition likelihoods even if valid action as there is always a PROB_OF_WRONG_XITION
                if valid_action:
                    valid_action_set.add(action_idx)
                else:
                    invalid_action_set.add(action_idx)
                #end else
            #end for loop through all actions of the domain (not just state)
            # ---now we have valid and invalid actions.
            for action_idx in invalid_action_set:
                mask_tensor[source_state_idx][action_idx] = 0.0
            #--end for
            action_potential_array *= mask_tensor

    #=====================================================================
    def compute_transition_likelihood_for_random_action(self, source_state_idx,possible_actions):
        """

        :param state_idx:
        :param possible_actions:
        :return:
        """
        transition_state_probabilities = torch.zeros(self.NUM_STATES)
        for other_action in possible_actions:
            target_state_idx = self.get_deterministic_next_state_with_move(source_state_idx, other_action)
            transition_state_probabilities[target_state_idx] += 1 / len(possible_actions)
            # we do += incase multiple actions lead to the same state
        # end for
        return transition_state_probabilities


    def get_s_s_s_confusion(self,m):
        m = np.array(m)
        num_states = m.shape[0]

        P = np.zeros((num_states, num_states, num_states))

        for s_star in range(num_states) : 
            for s_m in range(num_states) : 
                for salt in range(num_states) : 
                    P[s_star, s_m, salt] = (m[salt, s_m] + m[salt, s_star])/2

        return P 








