from src.policy_matrix_functions import *
from src.Combined_loss.config import *
import torch


def get_minmax_policy_values(minmax, all_params,env, policy_opt_lr = 1E-1, policy_grad_steps= 2000):
    """

    :param minmax:
    :return:
    """

    confusion_matrix_normalized,rsa_reward_matrix,prob_state,xition_prob_by_SxAxS2_matx = all_params
    action_potential_array_source = torch.randint(1, 10, rsa_reward_matrix.shape, dtype=torch.float32)
    noop_reward_cost_vector = torch.zeros(prob_state.shape)
    action_mask_invalid_actions = env.get_mask_for_invalid_actions()
    if REMOVE_INVALID_ACTIONS:
        action_potential_array_source *= action_mask_invalid_actions
    action_potential_array = Variable(action_potential_array_source, requires_grad=True)
    optimizer = torch.optim.RMSprop([action_potential_array], lr=policy_opt_lr)
    optimizer.zero_grad()
    prev_loss = -1
    curr_loss = 0  # just init values
    # ------------------------
    for i in range(policy_grad_steps):
        if abs(curr_loss - prev_loss) < LOSS_DELTA_CUTOFF:
            print("stopping after loss was less than cutoff = ", LOSS_DELTA_CUTOFF)
            break
        optimizer.zero_grad()
        # if abs(curr_loss-prev_loss) < STOPPING_LOSS_DELTA:
        #     print("change in error less than STOPPING_LOSS_DELTA, stopping search")
        #     break
        # ---end if
        prev_loss = curr_loss
        # todo try abs() instead of squared, compare performance
        with torch.no_grad():
            action_potential_array.clamp_(min=1, max=1e4)
        loss, loss_value_term, loss_complexity_term, loss_regularization_term, null_action_likelihood_vector, policy_matrix = \
            policy_gradient_step(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,
                                 rsa_reward_matrix, confusion_matrix_normalized, action_mask_invalid_actions,
                                 action_potential_array, loss_value=True, loss_complexity=False, loss_regulariz=False,
                                 min_value_possible=0, value_norm_denominator=1, min_complex_possible=0,
                                 complex_norm_denominator=1,complexity_weight =0,null_actions_enabled=False)
        loss *= minmax
        loss.backward()
        optimizer.step()
        curr_loss = loss.item()
        if i % 1000 == 0:
            print("at iteration =", i, " : loss = ", loss)
            print("loss_value_term = ", loss_value_term)
            print("==========")
    # end for loop
    opt_value = -1*loss_value_term.item() #*-1 to remove sign
    assoc_complexity = compute_policy_confusion_cost_soft_policy(prob_state, confusion_matrix_normalized, policy_matrix)
    if minmax == MINIMIZE: #it's the opposite sign for policy value, since we optimize -1*value
        print("END of computing to find the maximum policy value ", opt_value , " with true assoc_complexity = ",assoc_complexity)
        print("TRUE CONFUSION SCORE = ",
              compute_true_policy_confusion_score( confusion_matrix_normalized, policy_matrix))
    else:
        print("END of computing to find the minimum policy value ", opt_value, " with true assoc_complexity = ",
              assoc_complexity)
        print("TRUE CONFUSION SCORE = ",
              compute_true_policy_confusion_score( confusion_matrix_normalized, policy_matrix))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return opt_value

#==================================================================================================
def get_minmax_policy_complexity(minmax, all_params, env, policy_opt_lr = 1E-1, policy_grad_steps= 2000):
    """

    :param minmax:
    :return:
    """
    confusion_matrix_normalized,rsa_reward_matrix,prob_state,xition_prob_by_SxAxS2_matx = all_params
    action_potential_array_source = torch.randint(1, 10, rsa_reward_matrix.shape, dtype=torch.float32)
    noop_reward_cost_vector = torch.zeros(prob_state.shape)
    action_mask_invalid_actions = env.get_mask_for_invalid_actions()
    if REMOVE_INVALID_ACTIONS:
        action_potential_array_source *= action_mask_invalid_actions
    action_potential_array = Variable(action_potential_array_source, requires_grad=True)
    optimizer = torch.optim.RMSprop([action_potential_array], lr=policy_opt_lr)
    optimizer.zero_grad()
    prev_loss = -1
    curr_loss = 0  # just init values
    # ------------------------
    for i in range(policy_grad_steps):
        if abs(curr_loss - prev_loss) < LOSS_DELTA_CUTOFF:
            print("stopping after loss was less than cutoff = ", LOSS_DELTA_CUTOFF)
            break
        optimizer.zero_grad()
        # if abs(curr_loss-prev_loss) < STOPPING_LOSS_DELTA:
        #     print("change in error less than STOPPING_LOSS_DELTA, stopping search")
        #     break
        # ---end if
        prev_loss = curr_loss
        # todo try abs() instead of squared, compare performance
        with torch.no_grad():
            action_potential_array.clamp_(min=1, max=1e4)
        loss, loss_value_term, loss_complexity_term, loss_regularization_term, null_action_likelihood_vector, policy_matrix = \
            policy_gradient_step(prob_state, xition_prob_by_SxAxS2_matx, noop_reward_cost_vector,
                                 rsa_reward_matrix, confusion_matrix_normalized, action_mask_invalid_actions,
                                 action_potential_array, loss_value=False, loss_complexity=True, loss_regulariz=False,
                                 min_value_possible=0, value_norm_denominator=1, min_complex_possible=0,
                                 complex_norm_denominator=1,complexity_weight =1,null_actions_enabled=False)
        loss *= minmax
        loss.backward()
        optimizer.step()
        curr_loss = loss.item()
        if i % 1000 == 0:
            print("at iteration =", i, " : loss = ", loss)
            print("loss_complexity_term = ", loss_complexity_term)
            print("==========")
    # end for loop
    opt_complexity = loss_complexity_term.item() #*minmax to remove the sign
    assoc_complexity = compute_policy_confusion_cost_soft_policy(prob_state, confusion_matrix_normalized, policy_matrix)
    if minmax == MAXIMIZE:
        print("END of computing to find the maximum complexity value ", opt_complexity , " with true assoc_complexity = ",assoc_complexity)
        print("TRUE CONFUSION SCORE = ",
              compute_true_policy_confusion_score( confusion_matrix_normalized, policy_matrix))
    else:
        print("END of computing to find the minimum complexity value ", opt_complexity, " with true assoc_complexity = ",
              assoc_complexity)
        print("TRUE CONFUSION SCORE = ",
              compute_true_policy_confusion_score( confusion_matrix_normalized, policy_matrix))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return opt_complexity
