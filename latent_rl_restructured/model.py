import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import namedtuple

SavedInfo = namedtuple('SavedInfo', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self,
                 input_size=6,
                 hidden1_size=18,
                 hidden2_size=36,
                 action_size=3,
                 value_size=1):
        super(ActorCritic, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.LeakyReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.LeakyReLU(),
            nn.Linear(hidden2_size, hidden1_size),
            nn.LeakyReLU()
        )
        self.action = nn.Linear(hidden1_size, action_size)
        self.value = nn.Linear(hidden1_size, value_size)
        self.saved_info = []
        self.rewards = []

    def forward(self, x):
        x = self.fc(x)
        action_scores = self.action(x)
        x_value = self.value(x)
        return F.softmax(action_scores, dim=1), x_value

    def select_action(self, x):
        probs, state_value = self.forward(x)
        action_ = probs.multinomial()
        self.saved_info.append(SavedInfo(torch.log(probs.gather(1, Variable(action_.data))),
                                         state_value))
        return action_.data


class VAE(nn.Module):
    def __init__(self, input_size=6, hidden1_size=12, hidden2_size=4):
        super(VAE, self).__init__()

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
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.fc3(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return self.decode(z), mu, log_var


class CVAE(nn.Module):
    def __init__(self, input_size=6, class_size=1, hidden1_size=12, hidden2_size=4):
        super(CVAE, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Sequential(
            nn.Linear(input_size + class_size, hidden1_size),
            nn.ReLU()
        )

        self.fc21 = nn.Linear(hidden1_size, hidden2_size)
        self.fc22 = nn.Linear(hidden1_size, hidden2_size)

        self.fc3 = nn.Sequential(
            nn.Linear(hidden2_size + class_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, input_size),
        )

    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, x, z):
        return self.fc3(torch.cat([z, torch.index_select(x, 1, Variable(torch.LongTensor([self.input_size])))],
                                  dim=1))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return self.decode(x, z), mu, log_var
