import numpy as np
from sklearn.preprocessing import normalize


def init_transition_matrix(states, actions, normalize_as="uniform"):
    """
	Initializes a transition matrix and a flag to normalize the matrix. Eg. uniform normalization.
	:param states: states of the env
	:param actions: actions of the env
	:param normalize_as: Type of normalization.
	:return:
	"""
    m = len(states)
    n = len(actions)

    matrix = np.zeros((m, n, m))

    if normalize_as == "uniform":
        matrix.fill(1)

    matrix = normalize_transition_matrix(matrix)

    return matrix


def normalize_transition_matrix(matrix):
    """
	Normalizes a numpy matrix. For each state, the matrix[state] is normalized using scipy.
	:param matrix: transition matrix. numpy.
	:return: normalized matrix.
	"""
    # numpy matrix
    m = matrix.shape[0]
    for ix in range(m):
        matrix[ix] = normalize(matrix[ix], axis=1, norm='l1')

    return matrix


if __name__ == "__main__":
    matrix = init_transition_matrix([1, 2, 3], [1, 2])
    print(matrix)
