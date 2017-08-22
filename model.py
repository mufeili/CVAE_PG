"""
We referred to the examples provided by
https://github.com/pytorch/examples/tree/master/vae and
https://github.com/wiseodd/generative-models/tree/master/VAE/conditional_vae
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Value(nn.Module):
    """
    Define a value network for estimating state values.
    """
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(6, 600)
        self.fc2 = nn.Linear(600, 1)

    def forward(self, s):
        """

        Parameters
        ----------
        s: Variable whose data is Tensor of shape n x 6 with requires_grad=False
            state signal sent by the environment

        Returns
        -------
        v: Variable whose data is Tensor of shape n x 1 with requires_grad=True
            This estimates the state value at s.
        """
        return self.fc2(F.tanh(self.fc1(s)))


class Fnn(nn.Module):
    """
    Define a feedforward neural network which will be used as the feedforward policy.
    """
    def __init__(self):
        super(Fnn, self).__init__()
        self.fc1 = nn.Linear(6, 600)
        self.fc2 = nn.Linear(600, 3)

    def forward(self, s):
        """

        Parameters
        ----------
        s: Variable whose data is Tensor of shape 1 x 6 with requires_grad=False
            state signal sent by the environment

        Returns
        -------
        p: Variable whose data is Tensor of shape 1 x 3 with requires_grad=True
           The data of p may be interpreted as probability of taking discrete
           action 1, 2, 3 separately.
        """
        s_ = F.tanh(self.fc1(s))
        p = F.softmax(F.tanh(self.fc2(s_)))
        return p


class Cvae(nn.Module):
    """
    Define a CVAE model which will be used as the CVAE policy.
    """
    def __init__(self):
        super(Cvae, self).__init__()
        self.fc1 = nn.Linear(9, 500)
        self.fc21 = nn.Linear(500, 24)
        self.fc22 = nn.Linear(500, 24)
        self.fc3 = nn.Linear(30, 3)

    def encode(self, x):
        """

        Parameters
        ----------
        x: Variable whose data is Tensor of shape n x 9 with requires_grad=False
            x = (s, a) with a one-hot representation of a. Note when we are collecting
            real-time experience, where the corresponding actions are unknown, simply set
            a = [0, 0, 0].

        Returns
        -------
        mu_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            mean of the modeled gaussian distribution for h
        log_var_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            logarithm of the covariance of the modeled gaussian distribution for h
        """
        x_ = F.tanh(self.fc1(x))
        return F.tanh(self.fc21(x_)), F.tanh(self.fc22(x_))

    @staticmethod
    def reparameterize(mu_h, log_var_h):
        """
        Perform the reparametrization trick. The reparametrization trick in VAE
        makes it tractable for BP to improve the distribution of h.

        Parameters
        ----------
        mu_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            mean of the modeled gaussian distribution for h
        log_var_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            logarithm of the covariance of the modeled gaussian distribution for h

        Returns
        -------
        h_sampled: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            h_sampled = mu_h + epsilon x covariance, where epsilon is sampled from a
            standard normal distribution
        """
        std = log_var_h.mul(0.5).exp_()
        epsilon = th.Tensor(std.size()).normal_()
        epsilon = Variable(epsilon, requires_grad=False).type(th.Tensor)
        return epsilon.mul(std).add_(mu_h)

    def forward(self, x):
        """
        Reconstruct a given (s,a).

        Parameters
        ----------
        x: Variable whose data is Tensor of shape n x 9 with requires_grad=False
            x = (s, a) with a one-hot representation of a. Note when we are collecting
            real-time experience, where the corresponding actions are unknown, simply set
            a = [0, 0, 0].

        Returns
        -------
        p: Variable whose data is Tensor of shape n x 3 with requires_grad=True
           The data of p may be interpreted as probability of taking discrete
           action 1, 2, 3 separately.
        """
        mu_h, log_var_h = self.encode(x)
        indices = th.arange(0, 6).type(th.LongTensor)
        h = self.reparameterize(mu_h, log_var_h)
        p = F.softmax(F.tanh(self.fc3(th.cat([x.index_select(1, indices), h], 1))))
        # Note we return the mean and covariance of the modeled Gaussian and they will be
        # used to calculate the closed form KL divergence between the distribution of
        # p(h|s,a) and the distribution of p(h|s).
        return mu_h, log_var_h, p
