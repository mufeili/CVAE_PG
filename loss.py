import torch as th
from torch.autograd import Variable


def fnn_policy_loss(cumulative_return, value_estimated, prob_action, prob):
    """
    Calculate the loss function -\sum (Q-V(x))\log p(a|x) for the
    FNN policy.

    Parameters
    ----------
    cumulative_return: Variable whose data is Tensor of shape n x 1
        cumulative return following taking action a at state s
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
    return - th.sum((cumulative_return - value_estimated) * th.log(prob_action))\
           - th.sum(prob * th.log(prob))


def kl_divergence(mean1, log_var1, mean2, log_var2):
    """
    This calculates the KL divergence between multivariate Gaussian distributions
    N(mean1_i, cov1_i) and N(mean2_i, cov2_i), where i is the index for the pair.

    Parameters
    ----------
    mean1: Variable whose data is Tensor of shape n x 24
        the mean for N(mean1, cov1)
    log_var1: Variable whose data is Tensor of shape n x 24
        the logarithm of the covariance for N(mean1, cov1)
    mean2: Variable whose data is Tensor of shape n x 24
        the mean for N(mean2, cov2)
    log_var2: Variable whose data is Tensor of shape n x 24
        the logarithm of the covariance for N(mean2, cov2)

    Returns
    -------
    KL_divergences: Variable whose data is Tensor of shape n x 1
        the KL divergences between each pair of multivariate
        Gaussian distributions
    """
    # The KLE divergence between two multivariate Gaussian N(\mu_{1}, \Sigma_{1})
    # and N(\mu_{0}, \Sigma_{0}) is 0.5 * (tr(\Sigma_{0}^{-1}\Sigma_{1})+
    # (\mu_{0}-\mu_{1})^{T}\Sigma_{0}^{-1}(\mu_{0}-\mu_{1})-k+
    # \log(\det\Sigma_{0}/\det\Sigma_{1})
    cov1 = th.exp(log_var1)
    cov2 = th.exp(log_var2)
    cov2_inverse = Variable(th.ones(cov2.size()), requires_grad=True)/cov2
    return 0.5 * (th.sum(cov2_inverse * cov1 + (mean2 - mean1) * cov2_inverse * (mean2 - mean1), 1).view(-1, 1)
                  - mean1.size()[1] + th.log(th.sum(cov1, 1).view(-1, 1)/th.sum(cov2, 1).view(-1, 1)))


def cvae_policy_loss(cumulative_return, value_estimated, recon_prob_action,
                     prior_h_mean, prior_h_log_var, pos_h_mean, pos_h_log_var):
    """

    Parameters
    ----------
    cumulative_return: Variable whose data is Tensor of shape n x 1
        cumulative return following taking action a at state s
    value_estimated: Variable whose data is Tensor of shape n x 1
        estimated value from the value network at corresponding states
    recon_prob_action: Variable whose data is Tensor of shape n x 1
        reconstructed probability of taking the selected action at the
        corresponding state
    prior_h_mean: Variable whose data is Tensor of shape n x 24
        the mean of the Gaussian distribution for the prior h, i.e. h|s
    prior_h_log_var: Variable whose data is Tensor of shape n x 24
        the logarithm of the covariance matrix of the Gaussian distribution
        for the prior h, i.e. h|s
    pos_h_mean: Variable whose data is Tensor of shape n x 24
        the mean of the Gaussian distribution for the posterior h, i.e. h|s,a
    pos_h_log_var: Variable whose data is Tensor of shape n x 24
        the logarithm of the covariance matrix of the Gaussian distribution
        for the posterior h, i.e. h|s

    Returns
    -------
    loss: Variable whose data is Tensor of size 1
    """
    losses = - (cumulative_return - value_estimated) * (recon_prob_action - kl_divergence(pos_h_mean,
                                                                                          pos_h_log_var,
                                                                                          prior_h_mean,
                                                                                          prior_h_log_var))

    return th.sum(losses)
