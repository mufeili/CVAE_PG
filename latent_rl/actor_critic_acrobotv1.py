"""
Based on Pranjal Tandon's code: https://gist.github.com/pranz24/ba731e65e1b64bf3710159aa75736f90

small neural network trained using actor-critic in Pytorch
References:
David Silver's Lecture 7-Policy Gradient Methods: https://www.youtube.com/watch?v=KHZVXao4qXs&t=46s
Actor-critic example in Pytorch: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
"""
import argparse
import gym
import numpy as np
import time
from collections import namedtuple
from itertools import count
from util import ReplayBuffer


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch actor-critic for acrobot')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--buffer-capacity', type=int, default=10000, metavar='N',
                    help='capacity for the replay buffer')
args = parser.parse_args()


env = gym.make('Acrobot-v1')
print('observation_space: ', env.observation_space)
print('action_space: ', env.action_space)
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedInfo = namedtuple('SavedInfo', ['state', 'action', 'value'])


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


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=0.002)
buffer = ReplayBuffer(args.buffer_capacity)


def select_action(state_):
    state = torch.from_numpy(state_).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    action_ = probs.multinomial()
    model.saved_info.append(SavedInfo(state, action_, state_value))
    return action_.data


time_str = time.strftime("%Y%m%d-%H%M%S")
outdir = ''.join(['Acrobot-results_', time_str])
env = gym.wrappers.Monitor(env, outdir, force=True)


def finish_episode():
    R = 0
    saved_info = model.saved_info
    value_loss = 0
    cum_returns_ = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        cum_returns_.insert(0, R)
    cum_returns = torch.Tensor(cum_returns_)
    cum_returns = (cum_returns - cum_returns.mean()) / (cum_returns.std() + np.finfo(np.float32).eps)
    for (state, action, value), r in zip(saved_info, cum_returns):
        reward = r - value.data[0, 0]
        action.reinforce(reward)
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
        buffer.push(state, action, r)
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_info))
    gradients = [torch.ones(1)] + [None] * len(saved_info)
    autograd.backward(final_nodes, gradients)
    optimizer.step()
    del model.rewards[:]
    del model.saved_info[:]


running_reward = 10
for episode_no in count(1):
    state = env.reset()
    for t in range(1000): 
        action = select_action(state)
        state, reward, done, _ = env.step(action[0, 0])
        if args.render:
            env.render()
        model.rewards.append(reward)
        if done:
            break

    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()
    if episode_no % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            episode_no, t, running_reward))
    if episode_no > 160000:
        print("Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        env.close()
        break
