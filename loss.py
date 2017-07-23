import torch


def fnn_policy_loss(q_value, value_estimated, probabilities):
    """
    Calculate the loss function -\sum (Q-V(x))\log p(a|x) for the
    FNN policy.

    Parameters
    ----------
    q_value: Variable whose data is Tensor of shape n x 1
        empirical state-action values
    value_estimated: Variable whose data is Tensor of shape n x 1
        state values estimated
    probabilities: Variable whose data is Tensor of shape n x 1
        probabilities of taking the action selected

    Returns
    -------
    loss: Variable whose data is Tensor of size 1
    """
    return -torch.sum((q_value - value_estimated) *
                      torch.log(probabilities))
