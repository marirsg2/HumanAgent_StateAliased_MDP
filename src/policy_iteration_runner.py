"""

NEed to translate the policy matrix into state transitions, and no-op actions.
Also need the no-op reward put into the MRP.r(s) is the expected reward of that state given the policy, i.e. all the r(s,a) from that state
times the prob of p(s,a). The prob of p(s,s') is combined transition likelihood.

"""

import numpy as np
# from src.Combined_loss.minmax_normalization import get_minmax_policy_values
# from src.Combined_loss.minmax_normalization import get_minmax_complexity_values


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



# ENVNAME = args.env
# ENVNAME = "amazon"
# ENVNAME = None
# ENVNAME = "gridworld"
# ENVNAME = "colorworld"
# ENVNAME = "colorworld"



env = None 


# ENVNAME = args.env

# ENVNAME = "genericworld"
ENVNAME = "gridworld"

if ENVNAME == "gridworld" :
    env = GridWorldENV()
elif ENVNAME == "amazon":
    env = AmazonDomainWorld(default_env=True, use_existing=args.read)
elif ENVNAME == "colorworld":
    env = ColorWorldENV()
elif ENVNAME == "colorworld":
    env = ColorWorldENV()
else:
    # env = GridWorldENV()
    env = GenericDomainWorld()
    # env = AmazonDomainWorld(default_env=True, use_existing=False)



NUM_STATES = env.NUM_STATES
NUM_ACTIONS = env.NUM_ACTIONS


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# BEGINS HERE
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------


from src.policy_iteration import policy_improvement as pf


# pp = env.print_numpy_policy(p)

for i in range(20) : 

    seed = np.random.randint(0,100)
    p,v = pf(env, seed=seed)
    print (p[-1])
    pp = env.translate_and_print_policy(p)
    break

    


