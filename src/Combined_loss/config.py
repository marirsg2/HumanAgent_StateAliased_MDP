RANDOM_SEED = 0
PROB_OF_RANDOM_ACTION = 0.05
print("PROB_OF_RANDOM_ACTION",PROB_OF_RANDOM_ACTION)
NUM_STATES_PER_ROW = 3
NUM_STATES = NUM_STATES_PER_ROW**2
DISCOUNT_FACTOR =0.7
print("DISCOUNT_FACTOR",DISCOUNT_FACTOR)
NOOP_COST = -0.1#make sure it is NEGATIVE if you want to penalize inaction, which is the non-policy action from uncertainty
TRIALS_PER_SETTING = 30 # for the hill climbing search (SAPI)

#----------------------------IGNORE, DONT CHANGE ALL PARAMETERS BELOW THIS-------------------
#-------------------------
COMPUTATIONAL_TOLERANCE = 1E-10
EPSILON = 1e-20 #used in places where a small number is needed
LOSS_DELTA_CUTOFF = 1e-20

#-------------------
COMPLEXITY_INFLUENCER_SCALER_RANGE = [0.0]
L1_CONF_POTENTIALS_DIVISOR_RANGE = [5]
#----------------------------------------
#at least one of the following two must be true
INCLUDE_TRUE_POLICY_IN_SEARCH = False #leave these
INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH = True
ENABLE_1HOT_POLICY_REGULARIZATION = 0# 1 or 0 #leave this
if not ENABLE_1HOT_POLICY_REGULARIZATION:
    print("you have disabled ENABLE_1HOT_POLICY_REGULARIZATION")
WEIGHT_FOR_TRUE_POLICY_VALUE = 0.0 # leave this as is
INVALID_ACTION_PENALTY = -0 #IMPORTANT NOTE invalid actions will have their reward SET to this. Best leave it at zero
REMOVE_INVALID_ACTIONS = False #BETTER IF False 
MIN_MAX_TRIES = 1 #ignore
MINIMIZE = +1#ignore
MAXIMIZE = -1#ignore
#---------------

NUM_ACTIONS = 4 #up down left right. Action index 0,1,2,3 is up right, down left (clockwise)
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
#this is added to the policy matrix during branch and bound only.
ACTION_DELAY = 4

