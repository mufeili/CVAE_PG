import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    """
    Define a feedforward which will be used in the feedforward policy.
    """
    def __init__(self):
        super(FNN, self).__init__()
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
