from src.Combined_loss.config import * 
import sys
import os
import argparse
from domains.gridworld_domain.gridworld import GridWorldENV
from domains.amazon_domain.amazon_env_wrapper import AmazonDomainWorld
from domains.colorworld_domain.colorworld import ColorWorldENV
from domains.generic_domain.generic_env_wrapper import GenericDomainWorld

if not os.path.exists("./Results"):
    os.mkdir("./Results")


parser = argparse.ArgumentParser(description='Policy Minimization')
parser.add_argument('--env', type=str, default="gridworld", help="Env name can be g for gridworld or a for amazon")
parser.add_argument('-read', action="store_true", default="False", help="Read existing csv files for domains")
args = parser.parse_args()



ENVNAME = args.env



ENVNAME = "genericworld"

if ENVNAME == "gridworld" :
    env = GridWorldENV()
elif ENVNAME == "amazon":
    env = AmazonDomainWorld(default_env=True, use_existing=args.read)
elif ENVNAME == "colorworld":
    env = ColorWorldENV()
elif ENVNAME == "colorworld":
    env = ColorWorldENV()
else:
    env = GenericDomainWorld(use_existing=True)


NUM_STATES = env.NUM_STATES
NUM_ACTIONS = env.NUM_ACTIONS





print (env)









transition_matrix =  env.get_transition_matrix()

state_names = env.state_names.copy()
action_names = env.action_names.copy()



order_dict = {"l" : 3, "m" : 2, "s" : 1}


for s in state_names : 
    for a in action_names : 
        o1 = s.split("_")[0]
        o2 = a.split("_")[1]

        if order_dict[o1] > order_dict[o2] : 
            for s_ in state_names : 
                if s_ == s : 
                    env.update_transition_value(s, a, s_, 1, name=True)
                else : 
                    env.update_transition_value(s, a, s_, 0, name=True)


print (env.get_transition_matrix())


# env.write_transition_matrix(env.get_transition_matrix())









