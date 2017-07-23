import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    """
    Define a feedforward which will be used in the feedforward policy.
    """
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(6, 24)
        self.fc21 = nn.Linear(24, 1)
        self.fc22 = nn.Linear(24, 1)
        self.fc23 = nn.Linear(24, 1)

    def forward(self, s):
        """

        Parameters
        ----------
        s: Variable whose data is Tensor of shape 1 x 6 with requires_grad=False
            state signal sent by the environment

        Returns
        -------
        p1: Variable whose data is Tensor of shape 1 x 1 with requires_grad=True
            The data of p1 may be interpreted as probability of taking discrete
            action 1.
        p2: Same as p1, except that p2 corresponds to discrete action 2.
        p3: Same as p1, except that p3 corresponds to discrete action 3.

        """
        s_ = F.tanh(self.fc1(s))
        p1 = F.sigmoid(self.fc21(s_))
        p2 = F.sigmoid(self.fc22(s_))
        p3 = F.sigmoid(self.fc23(s_))
        p_sum = p1 + p2 + p3
        return p1 / p_sum, p2 / p_sum, p3 / p_sum


class Value(nn.Module):
    """
    Define a value network for estimating state values.
    """
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(6, 24)
        self.fc2 = nn.Linear(24, 1)

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
