"""
Based on Pranjal Tandon's code: https://gist.github.com/pranz24/ba731e65e1b64bf3710159aa75736f90

small neural network trained using actor-critic in Pytorch
References:
David Silver's Lecture 7-Policy Gradient Methods: https://www.youtube.com/watch?v=KHZVXao4qXs&t=46s
Actor-critic example in Pytorch: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
"""
import errno
import numpy as np
import os
import pickle
import time
from itertools import count

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from model import ActorCritic, SavedInfo
from util import Logger

args = get_args()


def main():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    print('time_str: ', time_str)

    exp_count = 0

    if args.experiment == 'a|s':
        direc_name_ = '_'.join([args.env, args.experiment])
    else:
        direc_name_ = '_'.join([args.env, args.experiment, 'bp2VAE', str(args.bp2VAE)])

    direc_name_exist = True

    while direc_name_exist:
        exp_count += 1
        direc_name = '/'.join([direc_name_, str(exp_count)])
        direc_name_exist = os.path.exists(direc_name)

    try:
        os.makedirs(direc_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if args.tensorboard_dir is None:
        logger = Logger('/'.join([direc_name, time_str]))
    else:
        logger = Logger(args.tensorboard_dir)

    env = gym.make(args.env)

    if args.wrapper:
        if args.video_dir is None:
            args.video_dir = '/'.join([direc_name, 'videos'])
        env = gym.wrappers.Monitor(env, args.video_dir, force=True)

    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.experiment == 'a|s':
        dim_x = env.observation_space.shape[0]
    elif args.experiment == 'a|z(s)' or args.experiment == 'a|z(s, s_next)':
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
        from util import ReplayBuffer, vae_loss_function

        vae = VAE()
        if args.use_cuda:
            vae.cuda()
        vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)

        if args.experiment == 'a|z(s)':
            from util import Transition_S2S as Transition
        elif args.experiment == 'a|z(s, s_next)':
            from util import Transition_S2SNext as Transition

        buffer = ReplayBuffer(args.buffer_capacity, Transition)

        update_vae = True

    if args.experiment == 'a|s':
        from util import Record_S
    elif args.experiment == 'a|z(s)':
        from util import Record_S2S
    elif args.experiment == 'a|z(s, s_next)' or args.experiment == 'a|z(a_prev, s, s_next)':
        from util import Record_S2SNext

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

            all_value_loss.append(value_loss.data.cpu()[0])
            all_policy_loss.append(policy_loss.data.cpu()[0])
        else:
            logger.scalar_summary('value_loss', value_loss.data[0], n)
            logger.scalar_summary('policy_loss', policy_loss.data[0], n)

            all_value_loss.append(value_loss.data[0])
            all_policy_loss.append(policy_loss.data[0])

        del policy.rewards[:]
        del policy.saved_info[:]

    if args.experiment != 'a|s':

        def train_vae(n):

            train_times = (n // args.vae_update_frequency - 1) * args.vae_update_times

            for i in range(args.vae_update_times):
                train_times += 1

                sample = buffer.sample(args.batch_size)
                batch = Transition(*zip(*sample))
                state_batch = torch.cat(batch.state)

                if args.experiment == 'a|z(s)':
                    recon_batch, mu, log_var = vae.forward(state_batch)

                    mse_loss, kl_loss = vae_loss_function(recon_batch, state_batch, mu, log_var, logger,
                                                          train_times, kl_discount=args.kl_weight,
                                                          mode=args.experiment)

                elif args.experiment == 'a|z(s, s_next)':
                    next_state_batch = Variable(torch.cat(batch.next_state),
                                                requires_grad=False)
                    predicted_batch, mu, log_var = vae.forward(state_batch)
                    mse_loss, kl_loss = vae_loss_function(predicted_batch, next_state_batch, mu, log_var,
                                                          logger, train_times, kl_discount=args.kl_weight,
                                                          mode=args.experiment)

                vae_loss = mse_loss + kl_loss

                vae_optimizer.zero_grad()
                vae_loss.backward()
                vae_optimizer.step()

                logger.scalar_summary('vae_loss', vae_loss.data[0], train_times)
                all_vae_loss.append(vae_loss.data[0])
                all_mse_loss.append(mse_loss.data[0])
                all_kl_loss.append(kl_loss.data[0])

                if len(all_vae_loss) > 200:
                    if abs(sum(all_vae_loss[-100:])/100 - sum(all_vae_loss[-200:-100])/100) < args.vae_update_threshold:
                        update_vae = False

    # To store cum_reward, value_loss and policy_loss from each episode
    all_cum_reward = []
    all_last_hundred_average = []
    all_value_loss = []
    all_policy_loss = []

    if args.experiment != 'a|s':
        # Store each vae_loss calculated
        all_vae_loss = []
        all_mse_loss = []
        all_kl_loss = []

    for episode in count(1):
        done = False
        state_ = torch.Tensor([env.reset()])
        cum_reward = 0

        while not done:
            if args.experiment == 'a|s':
                state = Variable(state_, requires_grad=False)
            elif args.experiment == 'a|z(s)' or 'a|z(s, s_next)':
                state_ = Variable(state_, requires_grad=False)
                mu, log_var = vae.encode(state_)

                if args.bp2VAE and update_vae:
                    state = vae.reparametrize(mu, log_var)
                else:
                    state = vae.reparametrize(mu, log_var).detach()

            action_ = policy.select_action(state)

            if args.use_cuda:
                action = action_.cpu()[0, 0]
            else:
                action = action_[0, 0]

            next_state_, reward, done, info = env.step(action)
            next_state_ = torch.Tensor([next_state_])
            cum_reward += reward

            if args.render:
                env.render()

            policy.rewards.append(reward)

            if args.experiment == 'a|z(s)':
                buffer.push(state_)
            elif args.experiment == 'a|z(s, s_next)':
                if not done:
                    buffer.push(state_, next_state_)

            state_ = next_state_

        train_actor_critic(episode)
        last_hundred_average = sum(all_cum_reward[-100:])/100

        logger.scalar_summary('cum_reward', cum_reward, episode)
        logger.scalar_summary('last_hundred_average', last_hundred_average, episode)

        all_cum_reward.append(cum_reward)
        all_last_hundred_average.append(last_hundred_average)

        if args.experiment != 'a|s' and episode % args.vae_update_frequency == 0 and update_vae:
            assert len(buffer) >= args.batch_size
            train_vae(episode)

        if episode % args.log_interval == 0:
            print('Episode {}\tLast cum return: {:5f}\t100-episodes average cum return: {:.2f}'.format(
                episode, cum_reward, last_hundred_average))

        if episode > args.num_episodes:
            print("100-episodes average cum return is now {} and "
                  "the last episode runs to {} time steps!".format(last_hundred_average, cum_reward))
            env.close()
            torch.save(policy, '/'.join([direc_name, 'model']))

            if args.experiment == 'a|s':
                record = Record_S(policy_loss=all_policy_loss, value_loss=all_value_loss,
                                  cum_reward=all_cum_reward, last_hundred_average=all_last_hundred_average)
            elif args.experiment == 'a|z(s)':
                record = Record_S2S(policy_loss=all_policy_loss, value_loss=all_value_loss,
                                    cum_reward=all_cum_reward, last_hundred_average=all_last_hundred_average,
                                    mse_recon_loss=all_mse_loss, kl_loss=all_kl_loss, vae_loss=all_vae_loss)
            elif args.experiment == 'a|z(s, s_next)' or args.experiment == 'a|z(a_prev, s, s_next)':
                record = Record_S2SNext(policy_loss=all_policy_loss, value_loss=all_value_loss,
                                        cum_reward=all_cum_reward, last_hundred_average=all_last_hundred_average,
                                        mse_pred_loss=all_mse_loss, kl_loss=all_kl_loss, vae_loss=all_vae_loss)

            pickle.dump(record, open('/'.join([direc_name, 'record']), 'wb'))

            break


if __name__ == "__main__":
    main()
