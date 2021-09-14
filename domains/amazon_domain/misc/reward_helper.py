import pandas as pd
import numpy as np


def init_reward_matrix(states, actions, DEFAULT, savepath):
    """
	Initializes a reward matrix using state names, action names, default reward value. Saves the reward matrix at the specified path.
    :param states: states of the env
    :param actions: actions of the env
    :param DEFAULT: Default value of reward
    :param savepath:
    :return:
    """
    m, n = len(states), len(actions)

    matrix = np.zeros((m, n))
    matrix.fill(DEFAULT)

    write_reward_matrix(states, actions, matrix, savepath)
    return matrix


def write_reward_matrix(states, actions, matrix, savepath):
    """
	saves a state x action matrix = reward matrix.
    :param states: states of the env
    :param actions: actions of the env
    :param matrix: takes the matrix to save
    :param savepath: saves a reward matrix at the savepath location.
    """
    df = pd.DataFrame(matrix, columns=actions, index=states, dtype=float)
    df.to_csv(savepath)


def read_reward_matrix(path):
    """

    :param path: reads a reward matrix from the specified path.
    :return: np matrix
    """
    df = pd.read_csv(path, header=None)

    # remove the action and state names.
    df.pop(df.columns[0])
    df = df.iloc[1:]
    df = df.astype(float)

    matrix = df.to_numpy()

    return matrix


if __name__ == "__main__":
    path = "../data/reward_matrix.csv"

    init_reward_matrix(states=["a", "b"], actions=["a1", "a2", "a3"], DEFAULT=10, savepath=path)
    read_reward_matrix(path)
