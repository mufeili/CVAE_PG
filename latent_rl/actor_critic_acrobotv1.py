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
from util import Logger, ReplayBuffer
from model import Policy


import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch actor-critic for acrobot')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--num-episodes', type=int, default=80000, metavar='N',
                    help='number of episodes to train')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--VAE-dir', type=str, default='VAE_kl_dim(z)=4_discount=3e-5',
                    help='directory for trained VAE model')
parser.add_argument('--wrapper', action='store_true', default=False)
parser.add_argument('--latent', action='store_true', default=False)
parser.add_argument('--reinforce', action='store_true', default=False)
parser.add_argument('--use-cuda', action='store_true', default=False)
args = parser.parse_args()

use_cuda = args.cuda and torch.cuda.is_available()

env = gym.make('Acrobot-v1')
print('observation_space: ', env.observation_space)
print('action_space: ', env.action_space)
env.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    Tensor = torch.cuda.FloatTensor
    torch.cuda.manual_seed_all(args.seed)
else:
    Tensor = torch.FloatTensor

SavedInfo = namedtuple('SavedInfo', ['state', 'log_prob', 'action', 'value'])

if args.latent:
    model = Policy(input_size=4, hidden1_size=12, hidden2_size=24)
    vae = torch.load(args.VAE_dir)
    vae.eval()
else:
    model = Policy()

if use_cuda:
    model.cuda()
    if args.latent:
        vae.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.002)


def select_action(state_):

    if args.latent:
        state_ = Variable(Tensor([state_]), requires_grad=False)
        mu, log_var = vae.encode(state_)
        state = vae.reparametrize(mu, log_var).data
    else:
        state = torch.from_numpy(state_).float().unsqueeze(0)

    probs, state_value = model(Variable(state))
    action_ = probs.multinomial()
    model.saved_info.append(SavedInfo(state, torch.log(probs.gather(1, Variable(action_.data))),
                                      action_, state_value))
    return action_.data


time_str = time.strftime("%Y%m%d-%H%M%S")
print('time_str: ', time_str)

if args.wrapper:
    outdir = ''.join(['Acrobot-results_', time_str])
    env = gym.wrappers.Monitor(env, outdir, force=True)

# Set the logger.
if args.reinforce:
    logger = Logger(''.join(['./', 'logs_actor_critic_reinforce', time_str]))
else:
    logger = Logger(''.join(['./', 'logs_actor_critic', time_str]))


def finish_episode(ep_number):
    R = 0
    saved_info = model.saved_info
    value_loss = 0

    if not args.reinforce:
        policy_loss = 0

    cum_returns_ = []

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        cum_returns_.insert(0, R)

    cum_returns = Tensor(cum_returns_)
    cum_returns = (cum_returns - cum_returns.mean()) \
                  / (cum_returns.std() + np.finfo(np.float32).eps)

    for (state, log_prob, action, value), r in zip(saved_info, cum_returns):
        reward = r - value.data[0, 0]

        if args.reinforce:
            action.reinforce(reward)
        else:
            policy_loss -= log_prob * reward

        value_loss += F.smooth_l1_loss(value, Variable(Tensor([r])))

    if use_cuda:
        logger.scalar_summary('value_loss', value_loss.data.cpu()[0], ep_number)
    else:
        logger.scalar_summary('value_loss', value_loss.data[0], ep_number)

    if not args.reinforce:
        if use_cuda:
            logger.scalar_summary('policy_loss', policy_loss.data.cpu()[0, 0], ep_number)
        else:
            logger.scalar_summary('policy_loss', policy_loss.data[0, 0], ep_number)
        total_loss = policy_loss + value_loss
        
    optimizer.zero_grad()

    if args.reinforce:
        final_nodes = [value_loss] + list(map(lambda p: p.action,
            saved_info))
        gradients = [torch.ones(1)] + [None] * len(saved_info)
        autograd.backward(final_nodes, gradients)
    else:
        total_loss.backward()

    optimizer.step()
    del model.rewards[:]
    del model.saved_info[:]


running_reward = 10
for episode_no in count(1):

    model.train()

    state = env.reset()

    for t in range(1000):

        if args.latent:
            if args.wrapper:
                action_ = select_action(env.env.env.state)
            else:
                action_ = select_action(env.env.state)
        else:
            action_ = select_action(state)

        if use_cuda:
            action = action_.cpu()[0, 0]
        else:
            action = action_[0, 0]

        state, reward, done, _ = env.step(action)

        if args.render:
            env.render()

        model.rewards.append(reward)

        if done:
            break

    model.eval()

    logger.scalar_summary('cum_return', t + 1, episode_no)

    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode(episode_no)

    if episode_no % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            episode_no, t, running_reward))

    if episode_no > args.num_episodes:
        print("Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        env.close()

        # Save the model for later use.
        torch.save(model, ''.join(['actor_critic_', time_str]))
        break
