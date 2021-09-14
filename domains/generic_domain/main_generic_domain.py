# Script to familiarize with the interface of GenericDomain

from domains.generic_domain.generic_env import Generic_Environment
from domains.generic_domain.generic_action import Action

# features = {"size": ["small", "large"], "fragile": [0, 1]}
# features = {"a" : [1], "b" : [0]}
# features_dict = {"state": features, "action": features}

env = Generic_Environment(features_dict=None, use_existing=True)
for s in env.states:
    print(s.name)

for a in env.actions:
    print(a.name)

print(env.get_reward_matrix())
print(env.get_transition_matrix())

# normalize False can result in invalid matrix -> make sure to either go with normalize True, or manually keep check or normalization.
# env.update_transition_value(0, 1, 0, 0.7, normalize=False, name=False)
# env.update_transition_value("st_0_small_0", "ac_2_large_0", "st_0_small_0", 0.7, normalize=False, name=True)

print(env.get_transition_matrix())

print ("FINALS")
print(env.states)
print(env.state_names)

print (env.get_confusion_matrix())