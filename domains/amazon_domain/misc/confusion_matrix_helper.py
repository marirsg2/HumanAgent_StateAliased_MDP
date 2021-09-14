import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

def init_confusion_matrix(states, DEFAULT, savepath):
    """
	Initializes a confusion matrix using state names, default confusion value. Saves the matrix at the specified path.
    :param states: states of the env
    :param DEFAULT: Default value of reward
    :param savepath:
    :return:
    """
    m = len(states)

    matrix = np.zeros((m, m))
    matrix.fill(DEFAULT)

    matrix = normalize_confusion_matrix(matrix)

    write_confusion_matrix(states, matrix, savepath)
    return matrix


def normalize_confusion_matrix(matrix):
    """
    :param matrix: transition matrix. numpy.
    :return: normalized matrix.
    """
    # numpy matrix
    matrix = normalize(matrix, axis=1, norm='l1')
    return matrix


def write_confusion_matrix(states, matrix, savepath):
    """
	saves a state x action matrix = reward matrix.
    :param states: states of the env
    :param actions: actions of the env
    :param matrix: takes the matrix to save
    :param savepath: saves a reward matrix at the savepath location.
    """
    df = pd.DataFrame(matrix, columns=states, index=states, dtype=float)
    df.to_csv(savepath)


def read_confusion_matrix(path):
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
    path = "../data/confusion_matrix.csv"

    # init_confusion_matrix(states=["a", "b", "c"], DEFAULT=1, savepath=path)
    m = read_confusion_matrix(path)

    print (m)

    
