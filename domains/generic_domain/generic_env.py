import domains.generic_domain.generic_action as generic_action
import domains.generic_domain.generic_state as generic_state

import pandas as pd
import os 

import domains.generic_domain.config_generic_domain_ as config

import domains.generic_domain.misc.helper as helper
import domains.generic_domain.misc.confusion_matrix_helper as confusion_matrix_helper
import domains.generic_domain.misc.reward_helper as reward_helper
import domains.generic_domain.misc.transition_matrix_helper as transition_matrix_helper
import domains.generic_domain.misc.state_action_name_helper as state_action_name_helper


class Generic_Environment:
    """
    Generic Environment Class. You can provide a specific feature_dict (like for amazon domain)
    or you can provide a list of states and actions in ./data/state_action_name.csv

    Run the class once to initialize everything by using use_existing=True
    Then run the class again after performing any required updates to the csv manually. 
    You can also use setters to update csv 

    You can also read / write / update the reward matirx, confusion matrix, transition matrix
    """
    def __init__(self, features_dict=None, use_existing=False,RANDOM_NOISE_RANGE=0):

        self.RANDOM_NOISE_RANGE = RANDOM_NOISE_RANGE

        # expecting a dict where key is the feature name and value is the list of all possible values of the feature.
        self.REWARD_MATRIX_PATH = config.REWARD_MATRIX_PATH
        self.CONFUSION_MATRIX_PATH = config.CONFUSION_MATRIX_PATH
        self.STATE_ACTION_NAME_PATH = config.STATE_ACTION_NAME_PATH
        self.TRANSITION_MATRIX_PATH = config.TRANSITION_MATRIX_PATH

        self.bookkeeping_reset()
        self.bookkeeping_init()
 
        


        if features_dict : 
            self.feature_space = features_dict["state"]
            self.action_space = features_dict["action"]

            self.actions = self.generate_actions()
            self.states = self.generate_states()
        else : 
            self.actions = self.via_action_names()
            self.states = self.via_state_names()


        self.state_names = self.get_state_names()
        self.action_names = self.get_action_names()

        self.statename2idx_map = generic_state.State.name_2_idx_map
        self.actionname2idx_map = generic_action.Action.name_2_idx_map


        # np matrices.
        self.transition_matrix = []
        self.reward_matrix = []
        self.confusion_matrix = []
        

        if use_existing:
            self.read_reward_matrix(self.REWARD_MATRIX_PATH)
            self.read_confusion_matrix(self.CONFUSION_MATRIX_PATH)
            self.read_transition_matrix(self.TRANSITION_MATRIX_PATH)
        else:
            self.initialize_reward_matrix()
            self.initialize_confusion_matrix()
            self.initialize_transition_matrix()

        assert len(self.states) > 0 and len(self.actions) > 0, "Generated zero actions or states"




    def set_potentials_divisor(self,divisor):
        pass #do nothing

    def bookkeeping_init(self):
        # make sure all the csv are initalized if required.

        if not os.path.exists(self.STATE_ACTION_NAME_PATH) : 
            # init state_action name 
            d = {"State" : [], "Action" : []} 
            df = pd.DataFrame(data=d)
            df.to_csv(self.STATE_ACTION_NAME_PATH, index=False)

            print ( self.STATE_ACTION_NAME_PATH, " init for the first time, populate it exiting now.")
            exit()





    def bookkeeping_reset(self):
        """
        Reset the classes State and Action
        """
        generic_state.State.reset()
        generic_action.Action.reset()

    def get_state_names(self):
        """

        :return: list of names of states.
        """
        states = []
        for s in self.states:
            states.append(s.name)
        return states

    def get_action_names(self):
        """

        :return: list of names of actions
        """
        actions = []
        for a in self.actions:
            actions.append(a.name)
        return actions


    def via_state_names(self, path=None):
        if path is None:
            path = self.STATE_ACTION_NAME_PATH
        x = state_action_name_helper.read_state_names(path)

        states = []
        for s in x:
            states.append(generic_state.State(s))
        return states

    def via_action_names(self, path=None):
        if path is None:
            path = self.STATE_ACTION_NAME_PATH
        x = state_action_name_helper.read_action_names(path)
        
        actions = []
        for a in x:
            actions.append(generic_action.Action(a))

        return actions


    def generate_actions(self):
        """

        :return: all possible actions.
        """
        x = self.action_space

        action_dicts = helper.generate_permutation_dicts(x)
        actions = []
        for a in action_dicts:
            actions.append(generic_action.Action(a))

        return actions

    def generate_states(self):
        """

        :return: all possible states.
        """
        x = self.feature_space

        state_dicts = helper.generate_permutation_dicts(x)
        states = []
        for s in state_dicts:
            states.append(generic_state.State(s))

        return states

    def get_transition_matrix(self):
        """

        :return: the transition matrix.
        """
        return self.transition_matrix

    def get_reward_matrix(self):
        """

        :return: Returns the reward matrix.
        """
        return self.reward_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def initialize_transition_matrix(self):
        """
        Initialize the transition matrix using a uniform distribution.
        """
        self.transition_matrix = transition_matrix_helper.init_transition_matrix(self.state_names, self.action_names, DEFAULT_TRANSITION_VALUE=config.DEFAULT_TRANSITION_VALUE)
        self.write_transition_matrix(self.transition_matrix)

    def write_transition_matrix(self, matrix, savepath=None):
        if savepath is None:
            savepath = self.TRANSITION_MATRIX_PATH
        transition_matrix_helper.write_transition_matrix(self.state_names, self.action_names, matrix, savepath)


    def read_transition_matrix(self, path=None) : 
        if path is None:
            path = self.TRANSITION_MATRIX_PATH
        self.transition_matrix = transition_matrix_helper.read_transition_matrix(   path=path, 
                                                                                    actions=self.action_names, 
                                                                                    actionname2idx=self.actionname2idx_map)


    def set_confusion_matrix_path(self, path):
        """
        Change reward path to this path.
        :param path: if present, this path is used.
        """
        self.CONFUSION_MATRIX_PATH = path

    def initialize_confusion_matrix(self, path=None):
        """
        Initialize reward matrix by a default value config.DEFAULT_REWARD_VALUE.
        :param path: if present, thi path is used.
        """
        if path is None:
            path = self.CONFUSION_MATRIX_PATH
        self.confusion_matrix = confusion_matrix_helper.init_confusion_matrix(self.state_names,
                                                              config.DEFAULT_CONFUSION_MATRIX_VALUE, path)
        if not os.path.exists(self.CONFUSION_MATRIX_PATH) : 
            self.write_confusion_matrix()

    def read_confusion_matrix(self, path):
        """
        Read reward from path csv.
        :param path: if present, this path is used. Default is specified in config.
        """
        if path is None:
            path = self.CONFUSION_MATRIX_PATH
        self.confusion_matrix = confusion_matrix_helper.read_confusion_matrix(path)

    def write_confusion_matrix(self, matrix, savepath=None):
        """
        :param matrix: matrix to write.
        :param savepath: if present, this path is used.
        """
        if savepath is None:
            path = self.CONFUSION_MATRIX_PATH
        confusion_matrix_helper.write_confusion_matrix(self.state_names, matrix, path)





    def set_reward_path(self, path):
        """
        Change reward path to this path.
        :param path: if present, this path is used.
        """
        self.REWARD_MATRIX_PATH = path

    def initialize_reward_matrix(self, path=None):
        """
        Initialize reward matrix by a default value config.DEFAULT_REWARD_VALUE.
        :param path: if present, thi path is used.
        """
        if path is None:
            path = self.REWARD_MATRIX_PATH
        self.reward_matrix = reward_helper.init_reward_matrix(self.state_names, self.action_names,
                                                              config.DEFAULT_REWARD_VALUE, path)
        if(self.RANDOM_NOISE_RANGE > 0) : 
            self.reward_matrix = reward_helper.make_noisy_reward_warehouse(m=self.reward_matrix, reward_noise_range=self.RANDOM_NOISE_RANGE)

        if not os.path.exists(self.REWARD_MATRIX_PATH) : 
            self.write_reward_matrix()

    def read_reward_matrix(self, path):
        """
        Read reward from path csv.
        :param path: if present, this path is used. Default is specified in config.
        """
        if path is None:
            path = self.REWARD_MATRIX_PATH
        self.reward_matrix = reward_helper.read_reward_matrix(path)
        self.reward_matrix = reward_helper.make_noisy_reward_warehouse(m=self.reward_matrix, reward_noise_range=self.RANDOM_NOISE_RANGE)

    def write_reward_matrix(self, matrix, savepath=None):
        """

        :param matrix: matrix to write.
        :param savepath: if present, this path is used.
        """
        if savepath is None:
            path = self.REWARD_MATRIX_PATH
        reward_helper.write_reward_matrix(self.state_names, self.action_names, matrix, path)

    def update_transition_value(self, s, a, s_, value, normalize=False, name=False):
        """
        Update transition matrix by sxaxs_ value.
        :param s: state
        :param a: action
        :param s_: state
        :param value: new value
        :param normalize: Whether to normalize the updated transition matrix.
        :param name: Whether s,a,s_ are names or indices.
        """
        if name:
            s = self.statename2idx_map[s]
            a = self.actionname2idx_map[a]
            s_ = self.statename2idx_map[s_]

        self.transition_matrix[s][a][s_] = value
        if normalize:
            self.normalize_transition_matrix()


    def normalize_transition_matrix(self) : 
        self.transition_matrix = transition_matrix_helper.normalize_transition_matrix(self.transition_matrix)

    def update_reward_matrix(self, s, a, value, name=False):
        """

        :param s: state
        :param a: action
        :param value: value to be changed
        :param name: Whether s,a are names or indices.
        """
        if name:
            s = self.statename2idx_map[s]
            a = self.actionname2idx_map[a]

        self.reward_matrix[s][a] = value

    def update_confusion_matrix(self, s, s_, value, name=False):
        """

        :param s: state
        :param s_: other state
        :param value: value to be changed
        :param name: Whether s,a are names or indices.
        """
        if name:
            s = self.statename2idx_map[s]
            s_ = self.actionname2idx_map[s_]

        self.confusion_matrix[s][s_] = value




if __name__ == "__main__":

    features = {"size": ["small", "large"], "fragile": [0, 1]}
    features_dict = {}

    features_dict["state"] = features
    features_dict["action"] = features
    print("IN ENV")
    env = Generic_Environment(features_dict)
    for s in env.states:
        print(s.name)

    for a in env.actions:
        print(a.name)
