from domains.amazon_domain import amazon_env
import torch

class AmazonDomainWorld (amazon_env.Amazon_Environment) : 
	def __init__(self, features_dict=None, use_existing= True, default_env=True) :

		if default_env : 
			features = {"size": ["small", "large"], "fragile": [0, 1]}
			features_dict = {}

			features_dict["state"] = features
			features_dict["action"] = features 

		super().__init__(features_dict,use_existing)

		self.NUM_STATES = len(self.states)
		self.NUM_ACTIONS = len(self.actions)


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
