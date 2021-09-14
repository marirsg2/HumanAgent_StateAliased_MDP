import numpy as np
import pandas as pd 
from sklearn.preprocessing import normalize


def init_confusion_matrix(states, DEFAULT_TRANSITION_VALUE=0, normalize_as="uniform"):
    """
	Initializes a transition matrix and a flag to normalize the matrix. Eg. uniform normalization.
	:param states: states of the env
	:param actions: actions of the env
	:param normalize_as: Type of normalization.
	:return:
	"""
    m = len(states)

    matrix = np.zeros((m, m, m))

    if normalize_as == "uniform":
        matrix.fill(DEFAULT_TRANSITION_VALUE)

    matrix = normalize_confusion_matrix(matrix)

    return matrix


def normalize_confusion_matrix(matrix):
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



def get_empty_str_matrix(n,m):
    mm = []
    for i in range(n) : 
        tmp = []
        for k in range(m):
            tmp.append("")
        mm.append(tmp)
    return mm 



def write_confusion_matrix(states, matrix, path):
    # use the matrix to populate this df 
    t = len(states)

    mm = get_empty_str_matrix(t, t)

    for s1, _ in enumerate(states) : 
        tmp = []

        for s2, _ in enumerate(states) : 
            val = ""
            for a, _ in enumerate(states) : 
                prob = matrix[s1][a][s2]
                if prob == 0 : 
                    continue 

                content = str(states[a]) + "@"  + str(prob)

                if val == "" : 
                    val += content
                else : 
                    val += ":" + content

            mm[s1][s2] = val

    df = pd.DataFrame(mm, columns=states, index=states, dtype=float)
    df.to_csv(path)

def read_confusion_matrix(path, states, statename2idx) : 
    df = pd.read_csv(path, header=None)
    df.pop(df.columns[0])
    df = df.iloc[1:]

    df = df.fillna("")

    mm = df.to_numpy()

    t = mm.shape[0]
    t_a = len(states)

    print (mm)

    matrix = np.zeros((t, t_a, t))

    for s1 in range(t) : 
        for s2 in range(t) : 
            val = mm[s1][s2]
            if val == '' : 
                continue

            x = val.split(":")
            for ix in x : 
                st = ix.split("@")
                this_state = statename2idx[st[0]]
                this_prob = float(st[1])

                matrix[s1][this_state][s2] = this_prob
    return matrix 







if __name__ == "__main__":
    import pprint 
    path = "../data/s_s_s_confusion_matrix.csv"


    states = ['a','b','c']
    name2idx = {'a' : 0, 'b' : 1, 'c' : 2}

    matrix = init_confusion_matrix(states, DEFAULT_TRANSITION_VALUE=1)

    # change some random index value.
    matrix[0][1][0] = 1
    print(matrix)

    write_confusion_matrix(states, matrix, path)

    r_matrix = read_confusion_matrix(path, states, name2idx)
    print (r_matrix)
    print (type(r_matrix))
    print ("Read same as write? ", np.array_equal(matrix, r_matrix))













