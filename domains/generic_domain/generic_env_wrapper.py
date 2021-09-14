from domains.generic_domain import generic_env
import torch
import numpy as np 

class GenericDomainWorld (generic_env.Generic_Environment) : 
	def __init__(self, features_dict=None, use_existing= True, default_env=False, read_state_action=False, RANDOM_NOISE_RANGE=0) :

		# for amazon env.
		if default_env : 
			features = {"size": ["small", "large"], "fragile": [0, 1]}
			features_dict = {}

			features_dict["state"] = features
			features_dict["action"] = features
		


		super().__init__(features_dict,use_existing,RANDOM_NOISE_RANGE)

		self.NUM_STATES = len(self.states)
		self.NUM_ACTIONS = len(self.actions)
		self.RANDOM_NOISE_RANGE = RANDOM_NOISE_RANGE

		self.NUM_STATES_PER_ROW = 'n/a'


	def compute_3d_sas2_xition_matrix(self):
		m = self.get_transition_matrix()
		m = torch.from_numpy(m)
		m = m.to(dtype= torch.float32)
		return m

	def get_reward_vector(self, invalid_action_penalty=False):
		m = self.get_reward_matrix()

		m = torch.from_numpy(m)
		m = m.to(dtype= torch.float32)
		return m

	def set_potentials_divisor(self, divisor):
		pass  # do nothing

	def get_confusion_potentials_matrix(self):
		m = self.get_confusion_matrix()
		m = torch.from_numpy(m)
		m = m.to(dtype= torch.float32)
		return m

	def translate_and_print_policy(self, policy_matrix, cutoff = 0.1):
		print ("Supposed to print Translate and Print policy")

	def remove_penalty(self, rsa_reward_vector):
		pass

	def get_mask_for_invalid_actions(self):
		mask_tensor = torch.ones([self.NUM_STATES, self.NUM_ACTIONS], dtype=torch.float32)
		return mask_tensor

	def get_s_s_s_confusion(self,m):
		m = np.array(m)
		num_states = m.shape[0]

		P = np.zeros((num_states, num_states, num_states))

		for s_star in range(num_states) : 
			for s_m in range(num_states) : 
				for salt in range(num_states) : 
					P[s_star, s_m, salt] = (m[salt, s_m] + m[salt, s_star])/2

		return P 
