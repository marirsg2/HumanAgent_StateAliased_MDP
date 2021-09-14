class Action:
    """
	Action class.
	"""
    counter = 0
    name_2_idx_map = {}

    def __init__(self, feature_dict):
        self.idx = Action.counter

        # action type can be made part of the feature_dict
        self.action = feature_dict

        Action.counter += 1

        # if features can dynamically change, need a callback to update self.name
        self.name = self.action_2_string()
        # Bookkeeping.
        Action.name_2_idx_map[self.name] = self.idx

    def action_2_string(self):
        """

		:return:get the name of the action.
		"""
        name = "ac_" + str(self.idx)
        for k in self.action.keys():
            name += "_" + str(self.action[k])

        return name

    @classmethod
    def reset(cls):
        """
		Reset the class variables.
		"""
        cls.counter = 0
        cls.name_2_idx_map = {}
