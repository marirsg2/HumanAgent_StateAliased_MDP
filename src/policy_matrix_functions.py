import torch
from src.Combined_loss.config import *
from torch.autograd import Variable
from torch import nn
import numpy as np
from src.value_iteration import *


#===========================================================================
def compute_xition_matrix(xition_prob_by_SxAxS2_matx,policy_matrix,null_action_likelihood_vector = None):
    """

    :param policy_matx:
    :return:

    policy is SxA
    the transition likelihoods is a 3d matrix SxAxS'
    So need to duplicate SxA in the 3rd dimension , then elementwise multiply and sum across dim=1 (2nd (action)dimension in 0-index)
    """
    try:
        if null_action_likelihood_vector == None:
            null_action_likelihood_vector = torch.zeros(policy_matrix.shape[0])
    except:
        pass #if it is a torch vector , cannot compare to None

    broadcasted_policy_matx = policy_matrix.view(policy_matrix.shape[0],policy_matrix.shape[1],1).repeat(1,1,policy_matrix.shape[0])
    xition_matx = torch.sum(xition_prob_by_SxAxS2_matx*broadcasted_policy_matx,dim=1)
        # Multiply all probabilities by (1-null prob), need to broadcast along rows and elementwise-multiply
    temp_non_null_action_likelihood = (1-null_action_likelihood_vector).view(1,null_action_likelihood_vector.shape[0]).repeat(xition_matx.shape[0],1).transpose(0,1)
    xition_matx = temp_non_null_action_likelihood*xition_matx
    xition_matx = xition_matx + torch.diag(null_action_likelihood_vector)
    return xition_matx

#===========================================================================

def compute_state_scores_for_bnb(classification_matrix,rsa_w_noop_cost,prob_state):
    """

    """
    max_abs_reward_by_state = np.max(np.abs(rsa_w_noop_cost),axis=1)
    broadcasted_max_abs_reward_by_state = np.tile(np.reshape(max_abs_reward_by_state,(1,prob_state.shape[0])),(classification_matrix.shape[0],1))
    broadcasted_prob_state = np.tile(np.reshape(prob_state,(1,prob_state.shape[0])),(classification_matrix.shape[0],1))

    score_vector = np.sum(classification_matrix*broadcasted_max_abs_reward_by_state*broadcasted_prob_state,axis = 1)
    # score_vector = np.sum(classification_matrix,axis = 1)
    return score_vector


#===========================================================================
def numpy_version_compute_xition_matrix(xition_prob_by_SxAxS2_matx,policy_matrix):
    """

    :param policy_matx:
    :return:

    policy is SxA
    the transition likelihoods is a 3d matrix SxAxS'
    So need to duplicate SxA in the 3rd dimension , then elementwise multiply and sum across dim=1 (2nd (action)dimension in 0-index)
    """


    broadcasted_policy_matx = np.tile(policy_matrix.reshape((policy_matrix.shape[0],policy_matrix.shape[1],1)),(1,1,policy_matrix.shape[0]))
    xition_matx = np.sum(xition_prob_by_SxAxS2_matx*broadcasted_policy_matx,axis=1)
    return xition_matx

#=====================================================================
# def old_compute_true_policy_confusion_score(confusion_matrix_normalized,policy_matrix):
#     """
#     :param prob_state_vector:
#     :param confusion_matrix_normalized:
#     :param policy_matrix:
#     :return:
#     :summary :
#
#
#         Sum_{s1 in S} [ (p(s1)) * Sum_{s2 in S} [ conf(s1,s2) *
#                                 Sum_{a in A} [ 1[max(p(s1)!= max(p(s2)]  ] ] ]
#
#         There is a cost of 1 , so long as the max prob action = the action we give to the human,
#         is not equal to the action in the other state
#
#     """
#     confusion_score = 0
#     num_states,num_actions = list(policy_matrix.shape)
#     for i in range(num_states):
#         state_i_action = torch.argmax(policy_matrix[i])
#         for j in range(num_states):
#             if state_i_action != torch.argmax(policy_matrix[j]):
#                 confusion_score += confusion_matrix_normalized[i][j]
#     return confusion_score/num_states #avg confusion

#=====================================================================
def compute_true_policy_confusion_score(confusion_matrix_normalized,alt_guess_3d_matrix,policy_matrix):
    """
    :param prob_state_vector:
    :param confusion_matrix_normalized:
    :param policy_matrix:
    :return:
    :summary :


        Sum_{s1 in S} [ (p(s1)) * Sum_{s2 in S} [ conf(s1,s2) *
                                Sum_{a in A} [ 1[max(p(s1)!= max(p(s2)]  ] ] ]

        There is a cost of 1 , so long as the max prob action = the action we give to the human,
        is not equal to the action in the other state

    """
    num_states,num_actions = list(policy_matrix.shape)

    #this function should only be called with a 1 hot policy
    check_for_1hot = torch.sum(torch.sqrt(policy_matrix))/num_states
    assert check_for_1hot == 1.0,"cannot COMPUTING TRUE CONFUSION SCORE with probabilistic policy, call function with 1hot policy"

    broadcasted_policy_matx = policy_matrix.view(policy_matrix.shape[0],
                                    policy_matrix.shape[1], 1).repeat(1, 1,policy_matrix.shape[0])
    xpose_broadcasted_policy_matx = torch.transpose(broadcasted_policy_matx,0,2)
    policy_dot_product = torch.sum( broadcasted_policy_matx* xpose_broadcasted_policy_matx,dim=1)
    #if policy is same we want it to be 0, and 1 otherwise. penalize dissimilarity
    policy_dissimil = (policy_dot_product-1)*-1 #this will invert the 0 and 1
    broadcasted_policy_dissimil_3d_matrix = policy_dissimil.view((policy_dissimil.shape[0],policy_dissimil.shape[1],1)).repeat((1,1,policy_dissimil.shape[0]))
    # but this needs to be duplicated along 2nd axis (axis index = 1), so we transpose dimensions 0 and 2.
    broadcasted_policy_dissimil_3d_matrix = torch.transpose(broadcasted_policy_dissimil_3d_matrix,0,2) # each horizontal slice has a SYMMETRIC s to s' policy disagreement matrix

    #compute all possible cases for confusion and store as a 3d matrix. This will be "convolved/crossed" with policy dissimilarity to get the prob of confusion
    probability_confusion_cases_3d_matrix = torch_compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(confusion_matrix_normalized,alt_guess_3d_matrix)
    #now we take the elementwise multiplication of the two 3d matrices, and sum along axis index 1,2 to get the delay probability for each state
    state_confusion_scores = torch.sum(broadcasted_policy_dissimil_3d_matrix*probability_confusion_cases_3d_matrix,dim=(1,2))
    #SAME as delay prob for that state !
    confusion_score = torch.sum(state_confusion_scores)/num_states #avg confusion
    return confusion_score
#=====================================================================
def compute_policy_confusion_cost_soft_policy(prob_state_vector, confusion_matrix_normalized, policy_matrix):
    """
    :param prob_state_vector:
    :param confusion_matrix_normalized:
    :param policy_matrix:
    :return:
    :summary :

    Sum_{s1 in S} [ (p(s1)) * Sum_{s2 in S} [ conf(s1,s2) *
                            Sum_{a in A} [ sqrt( (p(s1,a)-q(s2,a))^2 ] ] ]


    p(s1,a) is the correct policy's likelihood of choosing action "a"
    q(s2,a) is the confused state's policy's likelihood of choosing action "a"
    The last term is the confusion score because of the wrong action pair

    # There may not be a stationary distribution. so cannot say compute stationary distribution for the prob of states
    and use that for confusion matrix. Instead we just use the prob of starting state

    """

    # WAIT !!! the no-op reward/cost determines complexity effect, as gamma increases to 1.0.
    # If gamma is lower, that too favors complexity reduction.
    # Also if the COST(-ve) for no-op cost is increase, it will favor reduced complexity.

    num_states,num_actions = list(policy_matrix.shape)
    #tile prob state vector to be the same size as the confusion matrix (|S|x|S|)
    repeated_prob_state_vector = prob_state_vector.view(-1, 1).repeat(1, num_states)
    # compute p(s1)*conf(s1*s2)
    pS1_times_confS1S2 = repeated_prob_state_vector*confusion_matrix_normalized
    #----now compute the KL divergence part
    # EASIER to viz if we keep the second dimension as confused state, so we move the action dim to the 3rd dim
    p_matx = policy_matrix.view(num_states,1, num_actions).repeat(1,num_states,1)
    q_matx = torch.transpose(p_matx,0,1)#this switches s2 and s1. remember the actions are in the 3rd dim now (idx = 2)
    conf_scores_matrix = torch.sum(torch.sqrt( torch.pow(p_matx - q_matx + EPSILON,2)) , axis=2) #sum along the action dimension

    #WE ADD EPSILON because the gradient of sqrt(0) is nan.
    #finally combine the two parts and sum all the terms
    confusion_score = torch.sum(pS1_times_confS1S2*conf_scores_matrix)
    return confusion_score


#=====================================================================
def convert_policy_to_SA_xition_matrix(policy_matrix):
    """

    :param policy_matrix:
    :return:
    :summary: the policy is S -> A, we want S -> SA, so each row is expanded to |SxA|
    each original row's values is shifted to the appropriate subset of columns

    For our grid world, the invalid move actions are just treated as valid actions that return to the same state.
    #----
    TODO FOR LARGER SxA spaces, including invalid actions in SxA is very expensive (memory) and unnecessary.
     When a policy with confusion tries to execute an invalid action , we can just assign that invalid action likelihood
     of the soft policy to the state transition likelihood to itself = p(s_x,s_x). If there are state
     rewards in addition to r(s,a), we can just move the state rewards to all valid actions from that state.

    """
    S_to_SA_xition_matrix = torch.zeros((NUM_STATES,NUM_STATES*NUM_ACTIONS))
    for source_state_idx in range(NUM_STATES):
        start_col_idx = source_state_idx*NUM_ACTIONS
        end_col_idx = source_state_idx*NUM_ACTIONS + NUM_ACTIONS
        S_to_SA_xition_matrix[source_state_idx][start_col_idx:end_col_idx] = policy_matrix[source_state_idx]
    #---end for loop
    return S_to_SA_xition_matrix




# #=====================================================================
# def  DEPRECATED_compute_null_action_likelihoods_for_states_wNoopScaler(soft_policy_matrix, prob_state_confusion_matrix,noop_prob_scaler):
#     """
#     In comments and notation s_e is the state of the environment or true state.
#     The confusion matrix is read as follows. p(s_i|s_e) is the row index for s_i, and column index for s_e
#     """
#     assert noop_prob_scaler <= 1
#     assert noop_prob_scaler >= 0
#
#     norm_soft_policy_matrix = soft_policy_matrix/soft_policy_matrix.norm(2,dim=1,keepdim=True)
#     #we first compute (1- \hat{pi_{s_i} \dot \hat{pi_{s_j} ) . this is the soft disagreement between policies of s_i and s_j
#     soft_policy_matrix_size = list(norm_soft_policy_matrix.shape)
#     # We assume dim0 is the states, and dim 1 are the actions
#     # matrix_i = torch.tile(soft_policy_matrix,soft_policy_matrix_size+[soft_policy_matrix_size[0]])
#     matrix_i = norm_soft_policy_matrix.view(soft_policy_matrix_size + [1]).repeat([1, 1] + [soft_policy_matrix_size[0]])
#     matrix_j = torch.transpose(matrix_i,0,2)# this is to take the dot product
#     #----we also need the 1-hot penalty term for all pairs
#     # the reason for this will become clear soon
#     # temp_penalty_vector = torch.sum(torch.sqrt(policy_matrix),dim=1)
#     # penalty_matrix = temp_penalty_vector.view(-1,1).repeat(1,temp_penalty_vector.shape[0])
#     # pairwise_1hot_penalty = penalty_matrix*torch.transpose(penalty_matrix,0,1)
#     # soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1)/pairwise_1hot_penalty # we are taking the dot product for similarity, and also div by the penalty term. more penalty, more delay
#     soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1) # old way
#     # In the above computation We dont have to worry about negative similarities since the policy vector probabilities are all positives
#     confusion_in_hypoth_state = torch.matmul(soft_policy_dissimil,prob_state_confusion_matrix)# Human thinks it is s_i but
#     # pauses because there are other similar cases whose policy is different
#     # from the above step we did a matmul( s_i X s_j , s_j X s_e), so we are left with s_i x s_e matrix
#     # next we take the dot product with the prob of identifying a state as s_i , given the state is s_e. First prob_id is
#     prob_identification = prob_state_confusion_matrix #this is the likelihood of the state s_i being the MAP state in
#     # the human's mind. The likelihood that they settled on it. maybe bad attention, rush, other factors.
#
#     # prob_null_actions = noop_prob_scaler *torch.sum(prob_identification*confusion_in_hypoth_state, dim=0) # if we sum across the s_i
#     #--------------
#     prob_null_actions = torch.sum(prob_identification*confusion_in_hypoth_state, dim=0) # if we sum across the s_i
#     prob_null_actions_wScaling = prob_null_actions + (1-prob_null_actions)*noop_prob_scaler
#     # dimension, we are left with a vector over the s_e dimension. This is our null action likelihood
#     return prob_null_actions_wScaling
#     # return prob_null_actions

#=====================================================================
# TODO do nOT USE,  update to take in two matrices, confusion, and alternate guess matrix.
# def  compute_null_action_likelihoods_for_states_wDelayScaler(soft_policy_matrix, prob_state_confusion_matrix,noop_prob_scaler):
#     """
#     In comments and notation s_e is the state of the environment or true state.
#     The confusion matrix is read as follows. p(s_i|s_e) is the row index for s_i, and column index for s_e
#     """
#     assert noop_prob_scaler <= 1
#     assert noop_prob_scaler >= 0
#
#     norm_soft_policy_matrix = soft_policy_matrix/soft_policy_matrix.norm(2,dim=1,keepdim=True)
#     #we first compute (1- \hat{pi_{s_i} \dot \hat{pi_{s_j} ) . this is the soft disagreement between policies of s_i and s_j
#     soft_policy_matrix_size = list(norm_soft_policy_matrix.shape)
#     # We assume dim0 is the states, and dim 1 are the actions
#     # matrix_i = torch.tile(soft_policy_matrix,soft_policy_matrix_size+[soft_policy_matrix_size[0]])
#     matrix_i = norm_soft_policy_matrix.view(soft_policy_matrix_size + [1]).repeat([1, 1] + [soft_policy_matrix_size[0]])
#     matrix_j = torch.transpose(matrix_i,0,2)# this is to take the dot product
#     #----we also need the 1-hot penalty term for all pairs
#     # the reason for this will become clear soon
#     # temp_penalty_vector = torch.sum(torch.sqrt(policy_matrix),dim=1)
#     # penalty_matrix = temp_penalty_vector.view(-1,1).repeat(1,temp_penalty_vector.shape[0])
#     # pairwise_1hot_penalty = penalty_matrix*torch.transpose(penalty_matrix,0,1)
#     # soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1)/pairwise_1hot_penalty # we are taking the dot product for similarity, and also div by the penalty term. more penalty, more delay
#     soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1) # old way
#     # In the above computation We dont have to worry about negative similarities since the policy vector probabilities are all positives
#     confusion_in_hypoth_state = torch.matmul(soft_policy_dissimil,prob_state_confusion_matrix)# Human thinks it is s_i but
#     # pauses because there are other similar cases whose policy is different
#     # from the above step we did a matmul( s_i X s_j , s_j X s_e), so we are left with s_i x s_e matrix
#     # next we take the dot product with the prob of identifying a state as s_i , given the state is s_e. First prob_id is
#     prob_identification = prob_state_confusion_matrix #this is the likelihood of the state s_i being the MAP state in
#     # the human's mind. The likelihood that they settled on it. maybe bad attention, rush, other factors.
#
#     # prob_null_actions = noop_prob_scaler *torch.sum(prob_identification*confusion_in_hypoth_state, dim=0) # if we sum across the s_i
#     #--------------
#     prob_null_actions = torch.sum(prob_identification*confusion_in_hypoth_state, dim=0) # if we sum across the s_i
#     prob_null_actions_wScaling = prob_null_actions*noop_prob_scaler
#     # dimension, we are left with a vector over the s_e dimension. This is our null action likelihood
#     return prob_null_actions_wScaling
#     # return prob_null_actions

#=====================================================================
def numpy_compute_null_action_likelihoods_for_states_wDelayScaler(policy_matrix, prob_state_confusion_matrix, prob_alt_state_prob_matrix, noop_prob_scaler):
    """
        consider p(s-hat|s*) and p(s-alt|s-hat,s*)
        s-hat is the best guess of the human.
        s-alt is the best alternate guess of the human
        Notes: when the policy is partially defined this returns MIN prob of delay
    """
    assert noop_prob_scaler <= 1
    assert noop_prob_scaler >= 0
    #determine for which states the policy is defined. If it is a partial policy, then we assume that undefined states have matching policy, so the delay is lower
    policy_defined_states_vector = np.sum(policy_matrix, axis=1)
    broadcased_policy_UNdefined_states_2d_filter = np.tile(policy_defined_states_vector.reshape((policy_defined_states_vector.shape[0],1)),(1,policy_defined_states_vector.shape[0]))
    broadcased_policy_UNdefined_states_2d_filter = broadcased_policy_UNdefined_states_2d_filter * np.transpose(broadcased_policy_UNdefined_states_2d_filter,(1,0)) # So this is important
    # the above line is such that if the 2nd state is undefined, then we want to zero out the 2nd row, and 2nd column. This filter/mask is what the above line computes
    #we first compute (1- \hat{pi_{s_i} \dot \hat{pi_{s_j} ) . this is the soft disagreement between policies of s_i and s_j
    policy_matrix_size = list(policy_matrix.shape)
    # We assume dim0 is the states, and dim 1 are the actions
    matrix_i = np.tile(policy_matrix.reshape(policy_matrix_size + [1]), ([1, 1] + [policy_matrix_size[0]]))
    matrix_j = np.transpose(matrix_i, (2, 1, 0))# this is to take the elementwise product
    #----we also need the 1-hot penalty term for all pairs
    policy_dissimil = 1-np.sum(matrix_i*matrix_j,axis=1) # stores S,S' = 1.0 where the policy is dissimilar
    #zero out the principal diagonal, the policy cannot be dissimilar within the same state
    self_state_similarity_filter = np.ones(policy_dissimil.shape)
    np.fill_diagonal(self_state_similarity_filter,0)
    policy_dissimil *= self_state_similarity_filter #this is only necessary for partially defined policies, else it would naturally be the same
    #next zero out states where the policy is not yet defined. We assume it can be similar to get the pdmin. If policy is fully defined, this will be the true prob of delay
    policy_dissimil = policy_dissimil*broadcased_policy_UNdefined_states_2d_filter #if the policy is NOT defined for any state, then we consider that the policy matches in it's cases. zero out.
    broadcasted_policy_dissimil_3d_matrix = np.tile(policy_dissimil.reshape((policy_dissimil.shape[0],policy_dissimil.shape[1],1)),(1,1,policy_dissimil.shape[0]))
    # but this needs to be duplicated along 2nd axis (axis index = 1), so we transpose dimensions 0 and 2.
    broadcasted_policy_dissimil_3d_matrix = np.transpose(broadcasted_policy_dissimil_3d_matrix,(2,1,0))
    #compute all possible cases for confusion and store as a 3d matrix. This will be "convolved/crossed" with policy dissimilarity to get the prob of confusion
    probability_confusion_cases_3d_matrix = compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(prob_state_confusion_matrix,prob_alt_state_prob_matrix)
    #now we take the elementwise multiplication of the two 3d matrices, and sum along axis index 1,2 to get the delay probability for each state
    delay_prob_vector = np.sum(broadcasted_policy_dissimil_3d_matrix*probability_confusion_cases_3d_matrix,axis=(1,2))
    delay_prob_vector *= noop_prob_scaler #scale the probabilities
    return delay_prob_vector


#=====================================================================
def  torch_compute_null_action_likelihoods_for_states(soft_policy_matrix, prob_state_confusion_matrix, alt_guess_3d_matrix = None, noop_prob_scaler=1.0):
    """

    In comments and notation s_e is the state of the environment or true state.
    The confusion matrix is read as follows. p(s_i|s_e) is the row index for s_i, and column index for s_e
    """
    assert noop_prob_scaler <= 1
    assert noop_prob_scaler >= 0
    if alt_guess_3d_matrix is None:
        raise Exception("The old way of only 1 confusion matrix is gone, need to support the alternative guess matrix")
        #if you wanted a hack test, the following creates an alt guess matrix using the same probabilities as the confusion matrix
        # alt_guess_3d_matrix = copy.deepcopy(classification_matrix.cpu().detach().numpy())
        # alt_guess_3d_matrix = np.tile(np.reshape(alt_guess_3d_matrix.shape[0], classification_matrix.shape[1], 1),
        #                               (1, 1, classification_matrix.shape[0]))
        # alt_guess_3d_matrix = np.transpose(alt_guess_3d_matrix, (0, 2, 1))
        # alt_guess_3d_matrix = torch.from_numpy(alt_guess_3d_matrix)

    unitVec_soft_policy_matrix = soft_policy_matrix/soft_policy_matrix.norm(2,dim=1,keepdim=True)
    #we first compute (1- \hat{pi_{s_i} \dot \hat{pi_{s_j} ) . this is the soft disagreement between policies of s_i and s_j
    #also works for deterministic policies
    soft_policy_matrix_size = list(unitVec_soft_policy_matrix.shape)
    # We assume dim0 is the states, and dim 1 are the actions
    # matrix_i = torch.tile(soft_policy_matrix,soft_policy_matrix_size+[soft_policy_matrix_size[0]])
    matrix_i = unitVec_soft_policy_matrix.view(soft_policy_matrix_size + [1]).repeat([1, 1] + [soft_policy_matrix_size[0]])
    matrix_j = torch.transpose(matrix_i,0,2)# this is to take the dot product
    #----------------------
    soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1) # standard way
    broadcasted_policy_dissimil_3d_matrix = soft_policy_dissimil.view((soft_policy_dissimil.shape[0],soft_policy_dissimil.shape[1],1)).repeat((1,1,soft_policy_dissimil.shape[0]))
    # but this needs to be duplicated along 2nd axis (axis index = 1), so we transpose dimensions 0 and 2.
    broadcasted_policy_dissimil_3d_matrix = torch.transpose(broadcasted_policy_dissimil_3d_matrix,0,2) # each horizontal slice has a SYMMETRIC s to s' policy disagreement matrix

    #compute all possible cases for confusion and store as a 3d matrix. This will be "convolved/crossed" with policy dissimilarity to get the prob of confusion
    probability_confusion_cases_3d_matrix = torch_compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(prob_state_confusion_matrix,alt_guess_3d_matrix)
    #now we take the elementwise multiplication of the two 3d matrices, and sum along axis index 1,2 to get the delay probability for each state
    delay_prob_vector = torch.sum(broadcasted_policy_dissimil_3d_matrix*probability_confusion_cases_3d_matrix,dim=(1,2))
    delay_prob_vector *= noop_prob_scaler #scale the probabilities

    return delay_prob_vector
# #=====================================================================
def numpy_compute_null_action_likelihoods_for_states_wDelayScaler(policy_matrix, prob_state_confusion_matrix, prob_alt_state_prob_matrix, noop_prob_scaler):
    """
        consider p(s-hat|s*) and p(s-alt|s-hat,s*)
        s-hat is the best guess of the human.
        s-alt is the best alternate guess of the human
        Notes: when the policy is partially defined this returns MIN prob of delay
    """
    assert noop_prob_scaler <= 1
    assert noop_prob_scaler >= 0
    #determine for which states the policy is defined. If it is a partial policy, then we assume that undefined states have matching policy, so the delay is lower
    policy_defined_states_vector = np.sum(policy_matrix, axis=1)
    broadcased_policy_UNdefined_states_2d_filter = np.tile(policy_defined_states_vector.reshape((policy_defined_states_vector.shape[0],1)),(1,policy_defined_states_vector.shape[0]))
    broadcased_policy_UNdefined_states_2d_filter = broadcased_policy_UNdefined_states_2d_filter * np.transpose(broadcased_policy_UNdefined_states_2d_filter,(1,0)) # So this is important, read below
    # the above line is such that if the 2nd state is undefined, then we want to zero out the 2nd row, and 2nd column. This filter/mask is what the above line computes
    #we first compute (1- \hat{pi_{s_i} \dot \hat{pi_{s_j} ) . this is the soft disagreement between policies of s_i and s_j
    policy_matrix_size = list(policy_matrix.shape)
    # We assume dim0 is the states, and dim 1 are the actions
    matrix_i = np.tile(policy_matrix.reshape(policy_matrix_size + [1]), ([1, 1] + [policy_matrix_size[0]]))
    matrix_j = np.transpose(matrix_i, (2, 1, 0))# this is to take the elementwise product
    #----we also need the 1-hot penalty term for all pairs
    policy_dissimil = 1-np.sum(matrix_i*matrix_j,axis=1) # stores S,S' = 1.0 where the policy is dissimilar
    #zero out the principal diagonal, the policy cannot be dissimilar within the same state
    self_state_similarity_filter = np.ones(policy_dissimil.shape)
    np.fill_diagonal(self_state_similarity_filter,0)
    policy_dissimil *= self_state_similarity_filter #this is only necessary for partially defined policies, else it would naturally be the same
    #next zero out states where the policy is not yet defined. We assume it can be similar to get the pdmin. If policy is fully defined, this will be the true prob of delay
    policy_dissimil = policy_dissimil*broadcased_policy_UNdefined_states_2d_filter #if the policy is NOT defined for any state, then we consider that the policy matches in it's cases. zero out.
    broadcasted_policy_dissimil_3d_matrix = np.tile(policy_dissimil.reshape((policy_dissimil.shape[0],policy_dissimil.shape[1],1)),(1,1,policy_dissimil.shape[0]))
    # but this needs to be duplicated along 2nd axis (axis index = 1), so we transpose dimensions 0 and 2.
    broadcasted_policy_dissimil_3d_matrix = np.transpose(broadcasted_policy_dissimil_3d_matrix,(2,1,0))
    #compute all possible cases for confusion and store as a 3d matrix. This will be "convolved/crossed" with policy dissimilarity to get the prob of confusion
    probability_confusion_cases_3d_matrix = compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(prob_state_confusion_matrix,prob_alt_state_prob_matrix)
    #now we take the elementwise multiplication of the two 3d matrices, and sum along axis index 1,2 to get the delay probability for each state
    delay_prob_vector = np.sum(broadcasted_policy_dissimil_3d_matrix*probability_confusion_cases_3d_matrix,axis=(1,2))
    delay_prob_vector *= noop_prob_scaler #scale the probabilities
    return delay_prob_vector
#=====================================================================
#
def compute_complexity_loss(confusion_matrix_normalized,soft_policy_matrix):
    """

    """
    norm_soft_policy_matrix = soft_policy_matrix/soft_policy_matrix.norm(2,dim=1,keepdim=True)
    #we first compute (1- \hat{pi_{s_i} \dot \hat{pi_{s_j} ) . this is the soft disagreement between policies of s_i and s_j
    soft_policy_matrix_size = list(norm_soft_policy_matrix.shape)
    # We assume dim0 is the states, and dim 1 are the actions
    # matrix_i = torch.tile(soft_policy_matrix,soft_policy_matrix_size+[soft_policy_matrix_size[0]])
    matrix_i = norm_soft_policy_matrix.view(soft_policy_matrix_size + [1]).repeat([1, 1] + [soft_policy_matrix_size[0]])
    matrix_j = torch.transpose(matrix_i,0,2)# this is to take the dot product
    #----we also need the 1-hot penalty term for all pairs
    # the reason for this will become clear soon
    # temp_penalty_vector = torch.sum(torch.sqrt(policy_matrix),dim=1)
    # penalty_matrix = temp_penalty_vector.view(-1,1).repeat(1,temp_penalty_vector.shape[0])
    # pairwise_1hot_penalty = penalty_matrix*torch.transpose(penalty_matrix,0,1)
    # soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1)/pairwise_1hot_penalty # we are taking the dot product for similarity, and also div by the penalty term. more penalty, more delay
    soft_policy_dissimil = 1-torch.sum(matrix_i*matrix_j,dim=1) # old way
    # soft_policy_dissimil = 1-torch.sum(torch.pow(matrix_i,2)*torch.pow(matrix_j,2),dim=1) # old way
    # In the above computation We dont have to worry about negative similarities since the policy vector probabilities are all positives
    confusion_loss = torch.sum(confusion_matrix_normalized*soft_policy_dissimil)
    return confusion_loss

#=====================================================================
def compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, policy_matrix,prob_null_action,noop_reward_cost_vector):
    """

    Notes: Get prob of null action by the confusion likelihood
    """

    #update policy matrix likelihood by multiplying by project 1-prob_null, then append the null action likelihood as a column at the end
    prob_null_action_2d_view = prob_null_action.view(prob_null_action.shape[0],1)
    prob_null_tiled_matx = prob_null_action_2d_view.repeat(1,policy_matrix.shape[1]) #get the shape matrix the policy matrix. SxA
    updated_soft_policy = policy_matrix*(1-prob_null_tiled_matx)
    updated_soft_policy = torch.cat((prob_null_action_2d_view,updated_soft_policy),dim=1)
    #---
    #update the rsa reward matrix with no-op reward
    updated_rsa_matx = torch.cat((noop_reward_cost_vector.view(noop_reward_cost_vector.shape[0],1),rsa_reward_matrix),dim=1)
    expected_reward_vector = torch.sum(updated_rsa_matx*updated_soft_policy,dim=1)
    return expected_reward_vector
#=================================================================================

def policy_gradient_step(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,
                         rsa_reward_matrix, confusion_matrix_normalized, action_mask_invalid_actions,
                         action_potential_array, loss_value = True, loss_complexity = True, loss_regulariz = True,
                         min_value_possible=0, value_norm_denominator=1, min_complex_possible=0,
                         complex_norm_denominator=1, complexity_weight =0, null_actions_enabled = True,
                         weight_value = 1,weight_regularization =1):

    num_states, num_actions = list(rsa_reward_matrix.shape)

    scaled_potential_matx = action_potential_array
    # scaled_potential_matx = torch.softmax(action_potential_array, dim=1)

    # min_val, _ = torch.min(action_potential_array, dim=1, keepdim=True)
    # scaled_potential_matx = action_potential_array / min_val

    # max_val, _ = torch.max(action_potential_array, dim=1, keepdim=True)
    # scaled_potential_matx = torch.sqrt(action_potential_array / max_val)

    state_normaliz_denom = torch.sum(scaled_potential_matx, dim=1)
    # now tile it by the action set size
    state_normaliz_denom = state_normaliz_denom.view(-1, 1).repeat(1, num_actions)
    policy_matrix = scaled_potential_matx / state_normaliz_denom
    if REMOVE_INVALID_ACTIONS:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized,
                                            policy_matrix) * action_mask_invalid_actions
    else:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized, policy_matrix)
    loss_value_term = 0
    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    if loss_value:
        if null_actions_enabled:
            null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(policy_matrix,
                                                                                             confusion_matrix_normalized)
        # -----
        if INCLUDE_TRUE_POLICY_IN_SEARCH:
            s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, policy_matrix,
                                                                          null_action_likelihood_vector,
                                                                          noop_reward_cost_vector)  # we get the rsa added into the state reward
            prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, policy_matrix,
                                                     null_action_likelihood_vector)
            id_matx = torch.eye(num_states)
            state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                              s_reward_vector)
            # -----------------------
            scaled_state_value_vector = prob_state * state_value_vector
            loss_value_term +=  WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector))
            # loss_value_term += WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))


        # ---- also compute the value from the 1-hot policy got by taking the max operation on the policy
        if INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH:
            s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, confused_policy_matx,
                                                                          null_action_likelihood_vector,
                                                                          noop_reward_cost_vector)  # we get the rsa added into the state reward
            prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, confused_policy_matx,
                                                     null_action_likelihood_vector)
            # todo IMPORTANT add asserts equals for prob xition matrix. Simple mistake wasted a lot of time.
            id_matx = torch.eye(num_states)
            state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                              s_reward_vector)
            # -----------------------
            scaled_state_value_vector = prob_state * state_value_vector
            loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector))
            # loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))



        # ---end if
        loss_value_term /= (int(INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH) * (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) + \
                            WEIGHT_FOR_TRUE_POLICY_VALUE * int(
                    INCLUDE_TRUE_POLICY_IN_SEARCH))  # average of the two value functions.
        loss_value_term = -1 * weight_value*(loss_value_term - min_value_possible) / value_norm_denominator  # -1* to minimize

    # -------------------------
    loss_complexity_term = 0
    if loss_complexity:
        loss_complexity_term += compute_complexity_loss(confusion_matrix_normalized, policy_matrix)
        loss_complexity_term = (loss_complexity_term - min_complex_possible) / complex_norm_denominator
    #------------
    # loss_regularization_term = torch.sum(torch.sqrt(policy_matrix)) / max_regularization_loss #sometimes too weak
    loss_regularization_term = 0
    if loss_regulariz:
        loss_regularization_term += weight_regularization*torch.sum(
            torch.sqrt(policy_matrix)) / num_states  # THIS IS for stronger regularization
        # loss_regularization_term = torch.sum(torch.sqrt(policy_matrix))  #THIS IS for stronger regularization
    # -------------
    loss = (1-complexity_weight)*loss_value_term + complexity_weight * loss_complexity_term +\
                    ENABLE_1HOT_POLICY_REGULARIZATION * loss_regularization_term
    #-----------------
    return loss,loss_value_term,loss_complexity_term,loss_regularization_term,null_action_likelihood_vector,policy_matrix
#=================================================================================
def update_confusion_likelihoods(complexity_weight,confusion_matrix_normalized):
    """

    """
    above_zero_positions_1hot = confusion_matrix_normalized > 0
    above_zero_positions_1hot = above_zero_positions_1hot.to(dtype=torch.float32)
    uniform_probabilities = above_zero_positions_1hot/torch.sum(above_zero_positions_1hot,dim=1,keepdim=True)
    updated_identification_likelihoods = (1-complexity_weight)*confusion_matrix_normalized + complexity_weight*uniform_probabilities
    return updated_identification_likelihoods

#=================================================================================

def policy_gradient_step_noopProbScaler(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,noop_prob_scaler,
                         rsa_reward_matrix, confusion_matrix_normalized, action_mask_invalid_actions,
                         action_potential_array, loss_value = True, loss_complexity = True, loss_regulariz = True,
                         min_value_possible=0, value_norm_denominator=1, min_complex_possible=0,
                         complex_norm_denominator=1, complexity_weight =0, null_actions_enabled = True):

    num_states, num_actions = list(rsa_reward_matrix.shape)
    # scaled_potential_matx = torch.softmax(action_potential_array, dim=1)

    max_val, _ = torch.min(action_potential_array, dim=1, keepdim=True)
    scaled_potential_matx = torch.sqrt(action_potential_array / max_val)

    state_normaliz_denom = torch.sum(scaled_potential_matx, dim=1)
    # now tile it by the action set size
    state_normaliz_denom = state_normaliz_denom.view(-1, 1).repeat(1, num_actions)
    policy_matrix = scaled_potential_matx / state_normaliz_denom
    if REMOVE_INVALID_ACTIONS:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized,
                                            policy_matrix) * action_mask_invalid_actions
    else:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized, policy_matrix)
    loss_value_term = 0
    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    if loss_value:
        if null_actions_enabled:
            null_action_likelihood_vector = compute_null_action_likelihoods_for_states_wNoopScaler(policy_matrix,
                                                                                       confusion_matrix_normalized,noop_prob_scaler)
        # -----
        if INCLUDE_TRUE_POLICY_IN_SEARCH:
            s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, policy_matrix,
                                                                          null_action_likelihood_vector,
                                                                          noop_reward_cost_vector)  # we get the rsa added into the state reward
            prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, policy_matrix,
                                                     null_action_likelihood_vector)
            id_matx = torch.eye(num_states)
            state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                              s_reward_vector)
            # -----------------------
            scaled_state_value_vector = prob_state * state_value_vector
            # loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector))
            loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))


        # ---- also compute the value from the 1-hot policy got by taking the max operation on the policy
        if INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH:
            s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, confused_policy_matx,
                                                                          null_action_likelihood_vector,
                                                                          noop_reward_cost_vector)  # we get the rsa added into the state reward
            prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, confused_policy_matx,
                                                     null_action_likelihood_vector)
            # todo IMPORTANT add asserts equals for prob xition matrix. Simple mistake wasted a lot of time.
            id_matx = torch.eye(num_states)
            state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                              s_reward_vector)
            # -----------------------
            scaled_state_value_vector = prob_state * state_value_vector
            # loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector))
            loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))



        # ---end if
        loss_value_term /= (int(INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH) * (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) + \
                            WEIGHT_FOR_TRUE_POLICY_VALUE * int(
                    INCLUDE_TRUE_POLICY_IN_SEARCH))  # average of the two value functions.
        loss_value_term = -1 * (loss_value_term - min_value_possible) / value_norm_denominator  # -1* to minimize

    # -------------------------
    loss_complexity_term = 0
    if loss_complexity:
        loss_complexity_term += compute_complexity_loss(confusion_matrix_normalized, policy_matrix)
        loss_complexity_term = (loss_complexity_term - min_complex_possible) / complex_norm_denominator
    #------------
    # loss_regularization_term = torch.sum(torch.sqrt(policy_matrix)) / max_regularization_loss #sometimes too weak
    loss_regularization_term = 0
    if loss_regulariz:
        loss_regularization_term += torch.sum(
            torch.sqrt(policy_matrix)) / num_states  # THIS IS for stronger regularization
        # loss_regularization_term = torch.sum(torch.sqrt(policy_matrix))  #THIS IS for stronger regularization
    # -------------
    loss = loss_value_term + ENABLE_1HOT_POLICY_REGULARIZATION * loss_regularization_term
    #-----------------
    return loss,loss_value_term,loss_complexity_term,loss_regularization_term,null_action_likelihood_vector,policy_matrix


#=================================================================================

def iteration_value_computation(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,
                         rsa_reward_matrix, confusion_matrix_normalized,alt_guess_3d_matrix, action_mask_invalid_actions,
                         policy_matrix, null_actions_enabled = True):
    #----------------------------------
    num_states, num_actions = list(rsa_reward_matrix.shape)
    if REMOVE_INVALID_ACTIONS:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized,
                                            policy_matrix) * action_mask_invalid_actions
    else:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized, policy_matrix)

    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    if null_actions_enabled:
        null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(policy_matrix, confusion_matrix_normalized,alt_guess_3d_matrix,)
        # null_action_likelihood_vector = compute_null_action_likelihoods_for_states(policy_matrix,confusion_matrix_normalized)
    # -----
    value_term = torch.tensor(0,dtype=torch.float32)
    if INCLUDE_TRUE_POLICY_IN_SEARCH:
        s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, policy_matrix,
                                                                      null_action_likelihood_vector,
                                                                      noop_reward_cost_vector)  # we get the rsa added into the state reward
        prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, policy_matrix,
                                                 null_action_likelihood_vector)
        id_matx = torch.eye(num_states)
        state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                          s_reward_vector)
        # -----------------------
        scaled_state_value_vector = prob_state * state_value_vector
        value_term +=  WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector))
        # value_term += WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))


    # ---- also compute the value from the 1-hot policy got by taking the max operation on the policy
    if INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH:
        s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, confused_policy_matx,
                                                                      null_action_likelihood_vector,
                                                                      noop_reward_cost_vector)  # we get the rsa added into the state reward
        prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, confused_policy_matx,
                                                 null_action_likelihood_vector)
        # todo IMPORTANT add asserts equals for prob xition matrix. Simple mistake wasted a lot of time.
        id_matx = torch.eye(num_states)
        state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                          s_reward_vector)
        # -----------------------
        scaled_state_value_vector = prob_state * state_value_vector
        value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector))
        # value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))


    # ---end if
    value_term /= (int(INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH) * (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) + \
                        WEIGHT_FOR_TRUE_POLICY_VALUE * int(
                INCLUDE_TRUE_POLICY_IN_SEARCH))  # average of the two value functions.

    #-----------------
    return value_term
#=================================================================================

def policy_gradient_step_alternate_losses(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,
                         rsa_reward_matrix, confusion_matrix_normalized, action_mask_invalid_actions,
                         action_potential_array, loss_value = True, loss_complexity = True, loss_regulariz = True,
                         min_value_possible=0, value_norm_denominator=1, min_complex_possible=0,
                         complex_norm_denominator=1, complexity_weight =0, null_actions_enabled = True,
                         weight_value = 1,weight_regularization =1):

    num_states, num_actions = list(rsa_reward_matrix.shape)

    scaled_potential_matx = action_potential_array
    # scaled_potential_matx = torch.softmax(action_potential_array, dim=1)

    # min_val, _ = torch.min(action_potential_array, dim=1, keepdim=True)
    # scaled_potential_matx = action_potential_array / min_val

    # max_val, _ = torch.max(action_potential_array, dim=1, keepdim=True)
    # scaled_potential_matx = torch.sqrt(action_potential_array / max_val)

    state_normaliz_denom = torch.sum(scaled_potential_matx, dim=1)
    # now tile it by the action set size
    state_normaliz_denom = state_normaliz_denom.view(-1, 1).repeat(1, num_actions)
    policy_matrix = scaled_potential_matx / state_normaliz_denom
    if REMOVE_INVALID_ACTIONS:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized,
                                            policy_matrix) * action_mask_invalid_actions
    else:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized, policy_matrix)
    loss_value_term = torch.tensor(0,dtype=torch.float32)
    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    if loss_value:
        if null_actions_enabled:
            null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(policy_matrix,
                                                                                             confusion_matrix_normalized)
        # -----
        if INCLUDE_TRUE_POLICY_IN_SEARCH:
            s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, policy_matrix,
                                                                          null_action_likelihood_vector,
                                                                          noop_reward_cost_vector)  # we get the rsa added into the state reward
            prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, policy_matrix,
                                                     null_action_likelihood_vector)
            id_matx = torch.eye(num_states)
            state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                              s_reward_vector)
            # -----------------------
            scaled_state_value_vector = prob_state * state_value_vector
            loss_value_term +=  WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector))
            # loss_value_term += WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))


        # ---- also compute the value from the 1-hot policy got by taking the max operation on the policy
        if INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH:
            s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, confused_policy_matx,
                                                                          null_action_likelihood_vector,
                                                                          noop_reward_cost_vector)  # we get the rsa added into the state reward
            prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, confused_policy_matx,
                                                     null_action_likelihood_vector)
            # todo IMPORTANT add asserts equals for prob xition matrix. Simple mistake wasted a lot of time.
            id_matx = torch.eye(num_states)
            state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                              s_reward_vector)
            # -----------------------
            scaled_state_value_vector = prob_state * state_value_vector
            loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector))
            # loss_value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))



        # ---end if
        loss_value_term /= (int(INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH) * (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) + \
                            WEIGHT_FOR_TRUE_POLICY_VALUE * int(
                    INCLUDE_TRUE_POLICY_IN_SEARCH))  # average of the two value functions.
        loss_value_term = -1 * weight_value*(loss_value_term - min_value_possible) / value_norm_denominator  # -1* to minimize


    #------------
    # loss_regularization_term = torch.sum(torch.sqrt(policy_matrix)) / max_regularization_loss #sometimes too weak
    loss_regularization_term = torch.tensor(0,dtype=torch.float32)
    if loss_regulariz:
        # with torch.no_grad():
        #     worst_policy = torch.ones(policy_matrix.shape)/NUM_ACTIONS
        #     MAX_ENTROPY_LOSS = torch.sum(-1*torch.log2(worst_policy)* worst_policy)
        # loss_regularization_term += weight_regularization*torch.sum(torch.sqrt(policy_matrix)) / num_states  # THIS IS for stronger regularization
        # loss_regularization_term = torch.sum(torch.sqrt(policy_matrix))  #THIS IS for stronger regularization
        loss_regularization_term = torch.sum(-1*torch.log2(policy_matrix)* policy_matrix)#/MAX_ENTROPY_LOSS
        # -------------
    loss = loss_value_term + loss_regularization_term
    #-----------------
    return loss,loss_value_term,torch.tensor(0,dtype=torch.float32),loss_regularization_term,null_action_likelihood_vector,policy_matrix
#=================================================================================

def compute_expected_policy_value(confusion_matrix_normalized,
                                  rsa_reward_matrix,policy_matrix,
                                  null_action_likelihood_vector,noop_reward_cost_vector,xition_prob_by_SxAxS2_matx,prob_state):
    """

    """
    num_states, num_actions = list(policy_matrix.shape)
    confused_policy_matx = torch.matmul(confusion_matrix_normalized, policy_matrix)
    s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, confused_policy_matx,
                                                                  null_action_likelihood_vector,
                                                                  noop_reward_cost_vector)  # we get the rsa added into the state reward
    prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, confused_policy_matx,
                                             null_action_likelihood_vector)
    # todo IMPORTANT add asserts equals for prob xition matrix. Simple mistake wasted a lot of time.
    id_matx = torch.eye(num_states)
    state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                      s_reward_vector)
    # -----------------------
    scaled_state_value_vector = prob_state * state_value_vector
    return torch.sum(scaled_state_value_vector)

#=================================================================================

def numpy_compute_expected_policy_value(confusion_matrix_normalized,
                                  rsa_reward_matrix,policy_matrix,
                                  null_action_likelihood_vector,noop_reward_cost_vector,xition_prob_by_SxAxS2_matx,prob_state):
    """

    """
    confusion_matrix_normalized = torch.Tensor(confusion_matrix_normalized)
    rsa_reward_matrix = torch.Tensor(rsa_reward_matrix)
    policy_matrix = torch.Tensor(policy_matrix)
    null_action_likelihood_vector = torch.Tensor(null_action_likelihood_vector)
    noop_reward_cost_vector = torch.Tensor(noop_reward_cost_vector)
    xition_prob_by_SxAxS2_matx = torch.Tensor(xition_prob_by_SxAxS2_matx)
    prob_state = torch.Tensor(prob_state)
    return compute_expected_policy_value(confusion_matrix_normalized,
                                  rsa_reward_matrix,policy_matrix,
                                  null_action_likelihood_vector,noop_reward_cost_vector,xition_prob_by_SxAxS2_matx,prob_state)



#=================================================================================
def iteration_value_computation_inverted_values_sum(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,
                         rsa_reward_matrix, confusion_matrix_normalized,alt_guess_3d_matrix, action_mask_invalid_actions,
                         policy_matrix, null_actions_enabled = True):
    #----------------------------------
    num_states, num_actions = list(rsa_reward_matrix.shape)
    if REMOVE_INVALID_ACTIONS:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized,
                                            policy_matrix) * action_mask_invalid_actions
    else:
        confused_policy_matx = torch.matmul(confusion_matrix_normalized, policy_matrix)

    null_action_likelihood_vector = torch.zeros(prob_state.shape)
    if null_actions_enabled:
        null_action_likelihood_vector = torch_compute_null_action_likelihoods_for_states(policy_matrix, confusion_matrix_normalized, alt_guess_3d_matrix)
        # null_action_likelihood_vector = compute_null_action_likelihoods_for_states(policy_matrix,confusion_matrix_normalized)
    # -----
    value_term = torch.tensor(0,dtype=torch.float32)
    if INCLUDE_TRUE_POLICY_IN_SEARCH:
        s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, policy_matrix,
                                                                      null_action_likelihood_vector,
                                                                      noop_reward_cost_vector)  # we get the rsa added into the state reward
        prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, policy_matrix,
                                                 null_action_likelihood_vector)
        id_matx = torch.eye(num_states)
        state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                          s_reward_vector)
        # -----------------------
        scaled_state_value_vector = prob_state * state_value_vector
        scaled_state_value_vector = 1/(scaled_state_value_vector+1)#+1 over + epsilon so that after taking the average, it will only be [0,1] !
        value_term +=  WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector))
        # value_term += WEIGHT_FOR_TRUE_POLICY_VALUE* (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))


    # ---- also compute the value from the 1-hot policy got by taking the max operation on the policy
    if INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH:
        s_reward_vector = compute_equivalent_state_rewards_for_policy(rsa_reward_matrix, confused_policy_matx,
                                                                      null_action_likelihood_vector,
                                                                      noop_reward_cost_vector)  # we get the rsa added into the state reward
        prob_xition_matx = compute_xition_matrix(xition_prob_by_SxAxS2_matx, confused_policy_matx,
                                                 null_action_likelihood_vector)
        # todo IMPORTANT add asserts equals for prob xition matrix. Simple mistake wasted a lot of time.
        id_matx = torch.eye(num_states)
        state_value_vector = torch.matmul(torch.inverse(id_matx - DISCOUNT_FACTOR * prob_xition_matx),
                                          s_reward_vector)
        # -----------------------
        scaled_state_value_vector = prob_state * state_value_vector
        scaled_state_value_vector = 1/(scaled_state_value_vector+1) #+1 over + epsilon so that after taking the average, it will only be [0,1] !
        value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector))
        # value_term += (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) * (torch.mean(scaled_state_value_vector)+torch.min(scaled_state_value_vector))



    # ---end if
    value_term /= (int(INCLUDE_EXPECTED_CONFUSED_POLICY_IN_SEARCH) * (1 - WEIGHT_FOR_TRUE_POLICY_VALUE) + \
                        WEIGHT_FOR_TRUE_POLICY_VALUE * int(
                INCLUDE_TRUE_POLICY_IN_SEARCH))  # average of the two value functions.

    #-----------------
    return value_term

#=================================================================================
def torch_compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(confusion_matrix, alternate_guess_probabilities_3d_matrix):
    """
    Shat = S-hat , humans best guess :-)
    Salt = S-alt, humans best alternate guess

    """

    broadcasted_confusion_matx = confusion_matrix.view((confusion_matrix.shape[0],confusion_matrix.shape[1],1)).repeat((1,1,confusion_matrix.shape[0]))
    probability_confusion_cases_3d_matrix = broadcasted_confusion_matx * alternate_guess_probabilities_3d_matrix
    return probability_confusion_cases_3d_matrix

#=================================================================================
def compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(confusion_matrix, alternate_guess_probabilities_3d_matrix):
    """
    Shat = S-hat , humans best guess :-)
    Salt = S-alt, humans best alternate guess

    """
    broadcasted_confusion_matx = np.tile(confusion_matrix.reshape((confusion_matrix.shape[0],confusion_matrix.shape[1],1)),(1,1,confusion_matrix.shape[0]))
    probability_confusion_cases_3d_matrix = broadcasted_confusion_matx * alternate_guess_probabilities_3d_matrix
    return probability_confusion_cases_3d_matrix
    # return alternate_guess_probabilities_3d_matrix #Since the data in the alt guess matrix is for {s1,s2} to probability. Div by 2 is to avoid double counting symmetric cases

#=================================================================================
def compute_prob_delay_max(deterministic_partial_policy,confusion_matrix,alt_guess_3d_matrix,noop_prob_scaler=1.0):
    """

    """
    probability_confusion_cases_3d_matrix = compute_3d_matrix_probability_confusion_cases_S_Shat_Salt(confusion_matrix,alt_guess_3d_matrix)
    assert np.all(np.sum(probability_confusion_cases_3d_matrix, axis=(1, 2)) + COMPUTATIONAL_TOLERANCE > 1)
    #we need to zero out the cases where the policy matches, or KEEP the cases where the policy is DISSIMILAR
    policy_matrix_size = list(deterministic_partial_policy.shape)
    matrix_i = np.tile(deterministic_partial_policy.reshape(policy_matrix_size + [1]), ([1, 1] + [policy_matrix_size[0]]))
    matrix_j = np.transpose(matrix_i, (2, 1, 0))  # this is to take the elementwise product
    policy_dissimil = 1-np.sum(matrix_i * matrix_j, axis=1)  # stores S,S' = 1.0 where the policy is SIMILAR
    #zero out the principal diagonal, the policy cannot be dissimilar within the same state
    self_state_similarity_filter = np.ones(policy_dissimil.shape)
    np.fill_diagonal(self_state_similarity_filter,0)
    policy_dissimil *= self_state_similarity_filter# cannot be dissimilar with itself. this is only necessary for partially defined policies, else it would naturally be the same
    broadcasted_policy_dissimil = np.tile(policy_dissimil.reshape(list(policy_dissimil.shape) + [1]), (1,1, policy_matrix_size[0]))
    broadcasted_policy_dissimil = np.transpose(broadcasted_policy_dissimil,(2,1,0)) #even if the policy is undefined (zero) we consider that dissimilar, to get max prob dissimilarity
    #now we can zero out the cases where the prob matches
    probability_confusion_cases_3d_matrix = probability_confusion_cases_3d_matrix*broadcasted_policy_dissimil
    #max num of probability terms to keep per state (s*, 0 dimension index)
    list_for_factorial = list(range(1,max(policy_matrix_size[0],policy_matrix_size[1])+1))[abs(policy_matrix_size[0]-policy_matrix_size[1]):]
    max_num_prob_entries_for_delay = 1
    for x in list_for_factorial:
        max_num_prob_entries_for_delay *= x
    #---end for
    if max_num_prob_entries_for_delay > policy_matrix_size[0]**2: #
        max_num_prob_entries_for_delay = policy_matrix_size[0]**2
    #now compute pdmax by state
    p_dmax_list = []
    for state_idx in range(policy_matrix_size[0]):
        entries_for_state = np.ndarray.flatten(probability_confusion_cases_3d_matrix[state_idx])
        index_largest_prob_entries = np.argpartition(entries_for_state,-max_num_prob_entries_for_delay)[-max_num_prob_entries_for_delay:]
        p_dmax_for_state = np.sum(entries_for_state[index_largest_prob_entries])
        p_dmax_list.append(p_dmax_for_state)
    #---end for
    assert np.all(np.array(p_dmax_list)-COMPUTATIONAL_TOLERANCE < 1)
    return np.array(p_dmax_list)*noop_prob_scaler #still need to scale it



#=================================================================================
def compute_updated_RSA_and_SAS_with_delay(deterministic_partial_policy, original_rsa,original_sas, confusion_matrix, alt_guess_3d_matrix, delay_action_cost_vector,delay_action_effect, noop_prob_scaler= 1.0):
    """

    """
    p_dmin_vector = numpy_compute_null_action_likelihoods_for_states_wDelayScaler(deterministic_partial_policy, confusion_matrix, alt_guess_3d_matrix, noop_prob_scaler)
    p_dmax_array = compute_prob_delay_max(deterministic_partial_policy,confusion_matrix,alt_guess_3d_matrix,noop_prob_scaler)
    # print(p_dmin_vector)
    # print(p_dmax_array)
    #add null action to the sas cube. We assume for testing the null action just returns to the same state
    delay_action_effect =  np.transpose(np.atleast_3d(delay_action_effect),(0,2,1))
    original_sas_w_delay = np.append(original_sas,delay_action_effect, axis=1) # we want to tile on the diag matrix to the end of the action dimension
    # so the shape we want is 1xSxS.
    #the stochastic policy should INCLUDE the delay with prob P_dmin, and the other actions with (1-Pdmax)
    stochastic_partial_policy = np.matmul(confusion_matrix,deterministic_partial_policy)
    #multiply by each state's (1-p_dmax)
    #first match pdmax shape.
    p_dmax_array = np.tile(np.reshape(p_dmax_array,(p_dmax_array.shape[0],1)),(1,stochastic_partial_policy.shape[1]))
    stochastic_partial_policy *= (1-p_dmax_array)
    #add a column for the delay, with prob p_dmin
    stochastic_partial_policy = np.append(stochastic_partial_policy,np.transpose(np.atleast_2d(p_dmin_vector),(1,0)), axis=1)
    #-----------------------update SAS computation --------
    partial_policy_SxS2_matrix = numpy_version_compute_xition_matrix(original_sas_w_delay, stochastic_partial_policy)
    # convert this into an equivalent SxAxS2 matrix for merging with the original SAS2 matrix. First we make it SxS2xA, then take the transpose.
    fixed_SxAxS2 = np.tile(partial_policy_SxS2_matrix.reshape((partial_policy_SxS2_matrix.shape[0], partial_policy_SxS2_matrix.shape[1], 1)), (1, 1, stochastic_partial_policy.shape[1]))
    # take transpose to convert SxS2XA to SxAxS2
    fixed_SxAxS2 = np.transpose(fixed_SxAxS2,(0,2,1)) #this tells us that for all actions S->S2 with the same probability (for a fixed % of the time)
    # get the percentage of the policy that is defined
    fixed_policy_percentage_vector = np.sum(stochastic_partial_policy,axis=1)
    # match the dimensions of SAS2, such that each tile, in a vertical stack has the same probability. i.e. the first dimension is equal
    fixed_policy_percentage_SAS2_shape_matrix = np.tile(fixed_policy_percentage_vector.reshape((fixed_policy_percentage_vector.shape[0], 1, 1)),
                                                        (1, original_sas_w_delay.shape[1], original_sas_w_delay.shape[2]))
    #now combine the SAS2 matrices
    #NOTE we do not multiply the fixed_SxAxS2 with a ratio, because the stochastic_partial_policy matrix already considered the fixed ratio
    updated_SxAxS2 = fixed_SxAxS2 + (1-fixed_policy_percentage_SAS2_shape_matrix) * original_sas_w_delay
    #--------------UPDATED RSA computation -------------------
    original_rsa_w_delay = np.append(original_rsa,np.transpose(np.atleast_2d(delay_action_cost_vector),(1,0)),axis=1 )# we want to tile on the diag matrix to the end of the action dimension

    fixed_RSA = np.sum(original_rsa_w_delay*stochastic_partial_policy,axis=1) #the amount of reward that is fixed from the partial policy
    #the fixed RSA should be the same for all actions. So we need to sum along the actions axis, and then broadcast
    fixed_RSA = np.tile(fixed_RSA[:,np.newaxis],(1,original_rsa_w_delay.shape[1]))
    fixed_policy_percentage_SA_shape_matrix = np.tile(fixed_policy_percentage_vector[:,np.newaxis],(1,original_rsa_w_delay.shape[1]))
    updated_RSA = fixed_RSA + (1-fixed_policy_percentage_SA_shape_matrix)*original_rsa_w_delay
    return updated_RSA,updated_SxAxS2
#=================================================================================

if __name__ == "__main__":
    print("Testing partial policy to new SAS' matrix")
    #unit test 2 states, 3 actions
    deterministic_partial_policy = np.array([np.array([1, 0.0, 0.0]),np.array([0, 1, 0.0]), np.array([0, 0, 0])])
    original_sas = np.zeros((deterministic_partial_policy.shape[0],deterministic_partial_policy.shape[1],deterministic_partial_policy.shape[1]))
    original_sas[:,0,:] = 0.5#uniform probability of s' (0.5) for first action
    original_sas[:,1,0] = 1#second action always goes to state 1
    original_sas[:,2,1] = 1#third action always goes to state 2
    # print(original_sas)
    confusion_matrix = np.array([np.array([0.8, 0.1,0.1]),np.array([0.2, 0.6,0.2]),np.array([0.1, 0.2,0.7])])
    # alt_guess_3d_matrix = np.zeros((2,2))
    # np.fill_diagonal(alt_guess_3d_matrix,1)
    # alt_guess_3d_matrix = 1 - alt_guess_3d_matrix
    alt_guess_3d_matrix = np.ones((confusion_matrix.shape[0],confusion_matrix.shape[1]))*1/3 #sometimes confident in right answer, sometimes uncertain (like in color world)!
    alt_guess_3d_matrix = np.transpose(np.tile(alt_guess_3d_matrix.reshape(confusion_matrix.shape[0],confusion_matrix.shape[1],1),(1,1,confusion_matrix.shape[0])), (2,1,0))
    #add null action to the sas cube. We assume for testing the null action just returns to the same state
    delay_action_effect = np.zeros((original_sas.shape[0],original_sas.shape[0]))
    np.fill_diagonal(delay_action_effect,1)
    rsa_matrix = np.ones((original_sas.shape[0],original_sas.shape[1]))
    mod_rsa, modified_action_sas_w_delay = compute_updated_RSA_and_SAS_with_delay(deterministic_partial_policy,rsa_matrix, original_sas, confusion_matrix, alt_guess_3d_matrix, delay_action_effect, noop_prob_scaler=1.0)
    print(modified_action_sas_w_delay)

    state_set = list(range(deterministic_partial_policy.shape[0]))
    action_set = list(range(deterministic_partial_policy.shape[1]+1))#+1 for the delay action


    result = ValueIteration(state_set, action_set, rsa_matrix, modified_action_sas_w_delay, epsilon=0.001, gamma=0.9, ITERATION_CUTOFF=1000, d_start=None)
    print(result)
