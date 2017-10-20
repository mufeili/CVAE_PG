import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 18),
            nn.LeakyReLU(),
            nn.Linear(18, 36),
            nn.LeakyReLU(),
            nn.Linear(36, 18),
            nn.LeakyReLU()
        )
        self.action_head = nn.Linear(18, 3)
        self.value_head = nn.Linear(18, 1)
        self.saved_info = []
        self.rewards = []

    def forward(self, x):
        x = self.fc(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


class VAE(nn.Module):
    def __init__(self, input_size=4, hidden1_size=2, hidden2_size=1,
                 batch_norm=False):
        super(VAE, self).__init__()

        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bc1 = nn.BatchNorm1d(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU()
        )

        self.fc21 = nn.Linear(hidden1_size, hidden2_size)
        self.fc22 = nn.Linear(hidden1_size, hidden2_size)

        self.fc3 = nn.Sequential(
            nn.Linear(hidden2_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, input_size),
        )

    def encode(self, x):
        if self.batch_norm:
            x = self.bc1(x)
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.fc3(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
