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
from itertools import count
from model import ActorCritic, SavedInfo
from util import Logger

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str, default='a|z(s)',
                    help='''
                    four options:
                    1. a|s
                    2. a|z(s)
                    3. a|z(s, s_next)
                    4. a|z(a_prev, s, s_next)
                    ''')
parser.add_argument('--env', type=str, default='Acrobot-v1',
                    help='name of environment')
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
parser.add_argument('--wrapper', action='store_true', default=False)
parser.add_argument('--store-video', action='store_true', default=False)
parser.add_argument('--use-cuda', action='store_true', default=False)
parser.add_argument('--tensorboard-dir', type=str, default=None)
parser.add_argument('--video-dir', type=str, default=None)
parser.add_argument('--policy-lr', type=float, default=0.002,
                    help='learning rate for the policy model')

parser.add_argument('--z-dim', type=int, default=4,
                    help='dimensionality of bottleneck in VAE')
parser.add_argument('--vae-lr', type=float, default=5e-4,
                    help='learning rate for the vae model')
parser.add_argument('--buffer-capacity', type=int, default=50000, metavar='N',
                    help='capacity for the replay buffer')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training')
parser.add_argument('--vae-update-frequency', type=int, default=100, metavar='N',
                    help='update vae and empty the replay buffer every vae_update_frequency times')
parser.add_argument('--vae-update-times', type=int, default=10000, metavar='N',
                    help='update vae for vae_update_times every update')
parser.add_argument('--kl-divergence', action='store_true', default=False,
                    help='use kl divergence in the vae loss function or not')
parser.add_argument('--kl-weight', type=float, default=1,
                    help='hyperparameter that discounts kl divergence')
args = parser.parse_args()

args.use_cuda = args.use_cuda and torch.cuda.is_available()


def main():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    print('time_str: ', time_str)

    if args.tensorboard_dir is None:
        logger = Logger('_'.join([args.env, str(args.experiment), time_str]))
    else:
        logger = Logger(args.tensorboard_dir)

    env = gym.make(args.env)

    if args.wrapper:
        if args.video_dir is None:
            args.video_dir = '_'.join(['videos', args.env, str(args.experiment), time_str])
        env = gym.wrappers.Monitor(env, args.video_dir, force=True)

    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.experiment == 'a|s':
        dim_x = env.observation_space.shape[0]
    elif args.experiment == 'a|z(s)':
        dim_x = args.z_dim

    policy = ActorCritic(input_size=dim_x,
                         hidden1_size=3 * dim_x,
                         hidden2_size=6 * dim_x,
                         action_size=env.action_space.n)

    if args.use_cuda:
        Tensor = torch.cuda.FloatTensor
        torch.cuda.manual_seed_all(args.seed)
        policy.cuda()
    else:
        Tensor = torch.FloatTensor

    policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_lr)

    if args.experiment != 'a|s':
        from model import VAE
        from util import ReplayBuffer, Transition, vae_loss_function

        vae = VAE()
        if args.use_cuda:
            vae.cuda()
        vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)

        buffer = ReplayBuffer(args.buffer_capacity)

    def train_actor_critic(n):
        saved_info = policy.saved_info

        R = 0
        cum_returns_ = []

        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            cum_returns_.insert(0, R)

        cum_returns = Tensor(cum_returns_)
        cum_returns = (cum_returns - cum_returns.mean()) \
                      / (cum_returns.std() + np.finfo(np.float32).eps)
        cum_returns = Variable(cum_returns, requires_grad=False).unsqueeze(1)

        batch_info = SavedInfo(*zip(*saved_info))
        batch_log_prob = torch.cat(batch_info.log_prob)
        batch_value = torch.cat(batch_info.value)

        batch_adv = cum_returns - batch_value
        policy_loss = - torch.sum(batch_log_prob * batch_adv)
        value_loss = F.smooth_l1_loss(batch_value, cum_returns, size_average=False)

        policy_optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        policy_optimizer.step()

        if args.use_cuda:
            logger.scalar_summary('value_loss', value_loss.data.cpu()[0], n)
            logger.scalar_summary('policy_loss', policy_loss.data.cpu()[0], n)
        else:
            logger.scalar_summary('value_loss', value_loss.data[0], n)
            logger.scalar_summary('policy_loss', policy_loss.data[0], n)

        del policy.rewards[:]
        del policy.saved_info[:]

    if args.experiment == 'a|z(s)':

        def train_vae(n):

            train_times = (n // args.vae_update_frequency - 1) * args.vae_update_times

            for i in range(args.vae_update_times):
                train_times += 1

                sample = buffer.sample(args.batch_size)
                batch = Transition(*zip(*sample))

                state_batch = torch.cat(batch.state)
                recon_batch, mu, log_var = vae.forward(state_batch)
                vae_loss = vae_loss_function(recon_batch, state_batch, mu, log_var, logger,
                                             train_times, use_kl=args.kl_divergence, 
                                             kl_discount=args.kl_weight)
                vae_optimizer.zero_grad()
                vae_loss.backward()
                vae_optimizer.step()

                logger.scalar_summary('vae_loss', vae_loss.data[0], train_times)

    running_reward = 10

    for episode in count(1):
        done = False
        state_ = env.reset()
        cum_reward = 0

        while not done:
            if args.experiment == 'a|s':
                state = Variable(torch.from_numpy(state_).float().unsqueeze(0),
                                 requires_grad=False)
            elif args.experiment == 'a|z(s)':
                if args.wrapper:
                    state_ = env.env.env.state
                else:
                    state_ = env.env.state

                state_ = Variable(Tensor([state_]), requires_grad=False)
                mu, log_var = vae.encode(state_)
                state = Variable(vae.reparametrize(mu, log_var).data, requires_grad=False)

            action_ = policy.select_action(state)

            if args.use_cuda:
                action = action_.cpu()[0, 0]
            else:
                action = action_[0, 0]

            next_state_, reward, done, info = env.step(action)
            cum_reward += reward

            if args.render:
                env.render()

            policy.rewards.append(reward)

            if args.experiment == 'a|z(s)':
                buffer.push(state_)

            state_ = next_state_

        logger.scalar_summary('cum_return', cum_reward, episode)

        running_reward = running_reward * 0.99 + cum_reward * 0.01

        train_actor_critic(episode)
        if args.experiment == 'a|z(s)' and episode % args.vae_update_frequency == 0:
            train_vae(episode)
            buffer = ReplayBuffer(args.buffer_capacity)

        if episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5f}\tAverage length: {:.2f}'.format(
                episode, cum_reward, running_reward))

        if episode > args.num_episodes:
            print("Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, cum_reward))

            env.close()

            torch.save(policy, '_'.join(['actor_critic_', time_str]))
            break


if __name__ == "__main__":
    main()
