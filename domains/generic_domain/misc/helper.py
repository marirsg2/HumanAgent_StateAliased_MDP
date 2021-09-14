from pprint import pprint
from itertools import product as iter_product


def generate_permutation_dicts(x):
    """
	Takes a feature dictionary which contains all possible "features" and values of each feature.
	Input is like {"feature1" : [value1, value2...], "feature2" : ...}
	:param x: dict
	:return: list
	"""
    # x is feature_dict

    keys = x.keys()
    lists = []
    for k in keys:
        lists.append(x[k])

    states = []
    for r in iter_product(*lists):
        state = {}

        idx = 0
        for k in keys:
            state[k] = r[idx]
            idx += 1

        states.append(state)
    return states


if __name__ == "__main__":

    import sys

    sys.path.insert(0, "../")
    import amazon_state

    feature_dict = {"A": [0, 1], "B": [0, 1], "C": [0, 1, 2, 3]}


    def generate_states(x):
        state_dicts = generate_permutation_dicts(x)

        states = []
        for s in state_dicts:
            states.append(amazon_state.State(**s))

        return states


    states = generate_states(feature_dict)

    for s in states:
        print(s.name)
