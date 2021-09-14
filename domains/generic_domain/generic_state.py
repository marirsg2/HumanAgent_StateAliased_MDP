class State:
    """
	State Class for the Amazon Environment.

    feature_dict : if string, then treated as name of the state, if dict then its the feature dict.
    """
    counter = 0
    name_2_idx_map = {}

    def __init__(self, feature_dict):

        self.idx = State.counter
        self.state = feature_dict

        State.counter += 1

        # if features can dynamically change, need a callback to update self.name
        if  type(feature_dict) == type("x") : 
            self.name = feature_dict
        else : 
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
