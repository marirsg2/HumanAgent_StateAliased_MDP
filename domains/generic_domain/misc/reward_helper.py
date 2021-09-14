import pandas as pd
import numpy as np
import random

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

def make_noisy_reward_warehouse(m, reward_noise_range=0.2) : 
    # expect 2D reward matrix m

    # (Set the random seed before hand)
    assert abs(reward_noise_range) <= 1, "fix reward noise range to < 1, got " + str(reward_noise_range)

    random.seed(0)

    a, b = m.shape

    for i in range(a) : 
        for j in range(b) : 
            if(m[i,j] == 1 or m[i,j] == 0) : 
                continue 
            else : 
                # it will always be lower because of the -ve sign
                # it will be lower by alteast -0.1
                m[i,j] += -1*(random.random()*reward_noise_range + 0.1)

    return m

if __name__ == "__main__":
    import random 
    random.seed(2)
    
    path = "../data/reward_matrix.csv"

    # init_reward_matrix(states=["a", "b"], actions=["a1", "a2", "a3"], DEFAULT=10, savepath=path)
    r_m = read_reward_matrix(path)

    print (r_m)
    print (make_noisy_reward_warehouse(r_m))