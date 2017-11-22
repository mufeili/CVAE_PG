import argparse
import copy
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--check-interval', type=int, default=10, metavar='N',
                    help='interval between checks of different ways of update')
parser.add_argument('--check-times', type=int, default=10, metavar='N',
                    help='total times to check the difference between two ways of update')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedInfo = namedtuple('SavedInfo', ['action', 'value', 'log_prob'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_info = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores), state_values

    def select_action(self, state_):
        state_ = torch.from_numpy(state_).float().unsqueeze(0)
        probs, state_value = self.forward(Variable(state_, requires_grad=False))
        action = probs.multinomial()

        return action, state_value, probs


policy_reinforce = Policy()
optimizer_reinforce = optim.Adam(policy_reinforce.parameters(), lr=3e-2)

policy_bp = copy.deepcopy(policy_reinforce)
optimizer_bp = optim.Adam(policy_bp.parameters(), lr=3e-2)


def finish_episode():
    # r for values related to the model being updated via reinforce
    # b for values related to the model being updated via direct BP
    saved_info_r = policy_reinforce.saved_info
    value_loss_r = 0
    saved_info_b = policy_bp.saved_info
    value_loss_b = 0
    policy_loss_b = 0

    R = 0
    cum_returns = []

    for r in policy_reinforce.rewards[::-1]:
        R = r + args.gamma * R
        cum_returns.insert(0, R)

    cum_returns = torch.Tensor(cum_returns)
    cum_returns = (cum_returns - cum_returns.mean()) / (cum_returns.std() + np.finfo(np.float32).eps)

    for (action_r, value_r, log_prob_r), R in zip(saved_info_r, cum_returns):
        adv_r = R - value_r.data[0, 0]
        action_r.reinforce(adv_r)
        value_loss_r += F.smooth_l1_loss(value_r, Variable(torch.Tensor([R])))

    optimizer_reinforce.zero_grad()
    final_nodes = [value_loss_r] + list(map(lambda p: p.action, saved_info_r))
    gradients = [torch.ones(1)] + [None] * len(saved_info_r)
    autograd.backward(final_nodes, gradients)
    optimizer_reinforce.step()

    for (_, value_b, log_prob_b), R in zip(saved_info_b, cum_returns):
        adv_b = R - value_b.data[0, 0]
        policy_loss_b -= log_prob_b * Variable(torch.Tensor([[adv_b]]))
        # policy_loss_b -= log_prob_b * (1 - (1e-6/(Variable(log_prob_b.data.exp(), requires_grad=False) + 1e-6))) * adv_b
        value_loss_b += F.smooth_l1_loss(value_b, Variable(torch.Tensor([R])))

    optimizer_bp.zero_grad()
    total_loss_b = policy_loss_b + value_loss_b
    total_loss_b.backward()
    optimizer_bp.step()

    del policy_reinforce.rewards[:]
    del policy_reinforce.saved_info[:]
    del policy_bp.rewards[:]
    del policy_bp.saved_info[:]


def check_difference():
    reinforce_parameters = list(policy_reinforce.parameters())
    bp_parameters = list(policy_bp.parameters())
    difference = 0

    for i in range(len(reinforce_parameters)):
        difference += (reinforce_parameters[i] - bp_parameters[i]).norm(1).data[0]

    return difference


running_reward = 10
check_done = 0
differences = []
for i_episode in count(1):
    state = env.reset()
    done = False

    for t in range(10000): # Don't infinite loop while learning
        action_r, state_value_r, probs_r = policy_reinforce.select_action(state)
        policy_reinforce.saved_info.append(SavedInfo(action_r, state_value_r,
                                                     torch.log(probs_r.gather(1, Variable(action_r.data)))))

        _, state_value_b, probs_b = policy_bp.select_action(state)
        policy_bp.saved_info.append(SavedInfo(None, state_value_b,
                                              torch.log(probs_b.gather(1, Variable(action_r.data)))))

        state, reward, done, _ = env.step(action_r.data[0, 0])

        policy_reinforce.rewards.append(reward)
        policy_bp.rewards.append(reward)

        if done:
            break

    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()

    if i_episode % args.check_interval == 0:
        check_done += 1
        result = check_difference()
        differences.append(result)

    if check_done == args.check_times:
        print(differences)
        break
