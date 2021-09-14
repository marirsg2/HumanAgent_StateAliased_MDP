class State:
    """
	State Class for the Amazon Environment.
    """
    counter = 0
    name_2_idx_map = {}

    def __init__(self, feature_dict):
        self.idx = State.counter
        self.state = feature_dict

        State.counter += 1

        # if features can dynamically change, need a callback to update self.name
        self.name = self.state_2_string()
        # Bookkeeping.
        State.name_2_idx_map[self.name] = self.idx

    def state_2_string(self):
        """

        :return: name of the state
        """
        name = "st_" + str(self.idx)
        for k in self.state.keys():
            name += "_" + str(self.state[k])
        return name

    @classmethod
    def reset(cls):
        """

        """
        cls.counter = 0
        cls.name_2_idx_map = {}
