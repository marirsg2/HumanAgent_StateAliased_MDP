import pandas as pd
import numpy as np



def read_state_names(path):
    """

    :param path: reads a state/action matrix from the specified path.
    :return: np matrix
    """
    df = pd.read_csv(path)
    s_list = df['State'].dropna().tolist()

    return s_list

def read_action_names(path):
    """

    :param path: reads a state/action matrix from the specified path.
    :return: list of actions
    """
    df = pd.read_csv(path)
    a_list = df['Action'].dropna().tolist()

    return a_list



if __name__ == "__main__":
    path = "../data/state_action_matrix.csv"
    print (read_state_names(path))
    print (read_action_names(path))
