import domains.amazon_domain.amazon_action as amazon_action
import domains.amazon_domain.amazon_state as amazon_state

import pandas as pd

import domains.amazon_domain.config as config 

import domains.amazon_domain.misc.helper as helper
import domains.amazon_domain.misc.confusion_matrix_helper as confusion_matrix_helper
import domains.amazon_domain.misc.reward_helper as reward_helper
import domains.amazon_domain.misc.transition_matrix_helper as transition_matrix_helper


class Amazon_Environment:
    """
    Amazon Environment Class.
    """
    def __init__(self, features_dict, use_existing=True):
        self.bookkeeping_reset()

        # expecting a dict where key is the feature name and value is the list of all possible values of the feature.
        self.REWARD_MATRIX_PATH = config.REWARD_MATRIX_PATH
        self.CONFUSION_MATRIX_PATH = config.CONFUSION_MATRIX_PATH

        self.feature_space = features_dict["state"]
        self.action_space = features_dict["action"]

        self.actions = self.generate_actions()
        self.states = self.generate_states()

        self.state_names = self.get_state_names()
        self.action_names = self.get_action_names()

        # np matrices.
        self.transition_matrix = []
        self.reward_matrix = []
        self.confusion_matrix = []
        self.initialize_transition_matrix()

        if use_existing:
            self.read_reward_matrix(self.REWARD_MATRIX_PATH)
            self.read_confusion_matrix(self.CONFUSION_MATRIX_PATH)
        else:
            self.initialize_reward_matrix()
            self.initialize_confusion_matrix()

        assert len(self.states) > 0 and len(self.actions) > 0, "Generated zero actions or states"

        self.statename2idx_map = amazon_state.State.name_2_idx_map
        self.actionname2idx_map = amazon_action.Action.name_2_idx_map

    def bookkeeping_reset(self):
        """
        Reset the classes State and Action
        """
        amazon_state.State.reset()
        amazon_action.Action.reset()

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

    def generate_actions(self):
        """

        :return: all possible actions.
        """
        x = self.action_space

        action_dicts = helper.generate_permutation_dicts(x)
        actions = []
        for a in action_dicts:
            actions.append(amazon_action.Action(a))

        return actions

    def set_potentials_divisor(self,divisor):
        pass #do nothing

    def generate_states(self):
        """

        :return: all possible states.
        """
        x = self.feature_space

        state_dicts = helper.generate_permutation_dicts(x)
        states = []
        for s in state_dicts:
            states.append(amazon_state.State(s))

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
        self.transition_matrix = transition_matrix_helper.init_transition_matrix(self.state_names, self.action_names)





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
        confusion_matrix.write_confusion_matrix(self.state_names, matrix, path)





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

    def read_reward_matrix(self, path):
        """
        Read reward from path csv.
        :param path: if present, this path is used. Default is specified in config.
        """
        if path is None:
            path = self.REWARD_MATRIX_PATH
        self.reward_matrix = reward_helper.read_reward_matrix(path)

    def write_reward_matrix(self, matrix, savepath=None):
        """

        :param matrix: matrix to write.
        :param savepath: if present, this path is used.
        """
        if savepath is None:
            path = self.REWARD_MATRIX_PATH
        reward_helper.write_reward_matrix(self.state_names, self.action_names, matrix, path)

    def update_transition_value(self, s, a, s_, value, normalize=True, name=False):
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
    env = Amazon_Environment(features_dict)
    for s in env.states:
        print(s.name)

    for a in env.actions:
        print(a.name)
