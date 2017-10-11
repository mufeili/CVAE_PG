import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 18)
        self.affine2 = nn.Linear(18, 36)
        self.affine3 = nn.Linear(36, 18)
        self.action_head = nn.Linear(18, 3)
        self.value_head = nn.Linear(18, 1)
        self.saved_info = []
        self.rewards = []

    def forward(self, x):
        x = F.leaky_relu(self.affine1(x))
        x = F.leaky_relu(self.affine2(x))
        x = F.leaky_relu(self.affine3(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


class VAE(nn.Module):
    def __init__(self, input_size=4, hidden1_size=2, hidden2_size=1):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc21 = nn.Linear(hidden1_size, hidden2_size)
        self.fc22 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden1_size)
        self.fc4 = nn.Linear(hidden1_size, input_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
