import torch as th


def fnn_policy_loss(q_value, value_estimated, prob_action, prob):
    """
    Calculate the loss function -\sum (Q-V(x))\log p(a|x) for the
    FNN policy.

    Parameters
    ----------
    q_value: Variable whose data is Tensor of shape n x 1
        empirical state-action values
    value_estimated: Variable whose data is Tensor of shape n x 1
        state values estimated
    prob_action: Variable whose data is Tensor of shape n x 1
        probabilities of taking the action selected
    prob: Variable whose data is Tensor of shape n x 3
        probabilities of taking all actions conditioned on
        the state

    Returns
    -------
    loss: Variable whose data is Tensor of size 1
    """
    return - th.sum((q_value - value_estimated) * th.log(prob_action))\
           - th.sum(prob * th.log(prob))
