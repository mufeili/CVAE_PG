"""
We referred to the examples provided by
https://github.com/pytorch/examples/tree/master/vae and
https://github.com/wiseodd/generative-models/tree/master/VAE/conditional_vae
"""
import argparse
import errno
import os
import time


import torch as th
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


import gym


from model import Cvae, Value
from cvae_policy_train import cvae_policy_train


# Initialize an ArgumentParser instance for setting the conditions for
# a run from the terminal.
parser = argparse.ArgumentParser(description='Approach Acrobot-v1 with Policy\ '
                                             'Gradient using conditional\ '
                                             'Variational Autoencoder')
# The setting of metavar will be the displayed name for argument actions.

# The number of timesteps to run for each epoch is set to be 500 because the
# Acrobot-v1 task is set to end in 500 time steps in gym.
parser.add_argument('--update-frequency', type=int, default=5, metavar='N',
                    help='Update the network parameters every args.update_frequency /'
                         'episodes.')
parser.add_argument('--value-update-times', type=int, default=1, metavar='N',
                    help='Times to update a value network when it is the time for update')
parser.add_argument('--episodes', type=int, default=10000, metavar='N',
                    help='number of episodes for training')
parser.add_argument('--lr-cvae', type=float, default=1e-3, metavar='N',
                    help='learning rate for optimizing CVAE policy')
parser.add_argument('--lr-value', type=float, default=1e-2, metavar='N',
                    help='learning rate for optimizing value network')
parser.add_argument('--gamma', type=float, default=1, metavar='N',
                    help='discounting factor for the cumulative return')
parser.add_argument('--test-time', type=float, default=5, metavar='N',
                    help='number of times for test')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# If the option is specified, set args.no_cuda = True, otherwise it
# is False by default.
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--directory-name', type=str, default='/acrobot-results/cvae',
                    metavar='D', help='directory for storing results (default: '
                                      '/acrobot-results/cvae)')
parser.add_argument('--record', action='store_true', default=True,
                    help='enables recording gym results')
args_ = parser.parse_args()
args_.cuda = args_.use_cuda and th.cuda.is_available()


if args_.cuda:
    th.cuda.manual_seed(args_.seed)
    FloatTensor = th.cuda.FloatTensor
else:
    th.manual_seed(args_.seed)
    FloatTensor = th.FloatTensor
    import matplotlib.pyplot as plt
    plt.ion()
    try:
        os.makedirs('cvae/test')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
Tensor = FloatTensor


def main(args):
    policy = Cvae()
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr_cvae)
    value_network = Value()
    value_optimizer = optim.Adam(value_network.parameters(), lr=args.lr_value)
    mse_loss = nn.MSELoss()

    env = gym.make('Acrobot-v1')

    time_str, trained_model = cvae_policy_train(env,
                                                policy,
                                                value_network,
                                                policy_optimizer,
                                                value_optimizer,
                                                mse_loss,
                                                args)

    # Test the trained model using argmax.
    env = gym.make('Acrobot-v1')
    if args.record:
        # Store results from different runs of the model separately.
        results_directory = ''.join(['/tmp', args.directory_name, '/test/', time_str, '_discounting_',
                                     str(args.gamma), '_update_frequency_', str(args.update_frequency),
                                     '_value_update_times_', str(args.value_update_times)])
        env = gym.wrappers.Monitor(env, results_directory)

    for i in range(args.test_time):

        test_returns = []
        time_str = time.strftime("%Y%m%d-%H%M%S")

        state_ = env.reset()
        done = False
        cumulative_return = 0

        for timestep in range(0, 500):
            if not done:
                if not args.cuda:
                    env.render()
                state_ = th.from_numpy(state_.reshape(1, -1))
                state = Variable(state_, requires_grad=False).type(Tensor)
                padding = Variable(th.zeros(1, 3), requires_grad=False).type(Tensor)
                state_padded = th.cat([state, padding], 1)
                _, _, p = trained_model.forward(state_padded)
                action = th.max(p, 1)[1].data[0]
                next_state_, reward_, done, info_ = env.step(action)
                cumulative_return += (args.gamma ** timestep) * reward_
                state_ = next_state_

        print('====> Cumulative return: {}'.format(cumulative_return))

        plt.clf()
        plt.figure(2)
        plt.xlabel('episodes')
        plt.ylabel('cumulative returns')
        plt.plot(test_returns)
        plt.show()
        plt.savefig(''.join(['cvae/test/', time_str, '_discounting_',
                             str(args.gamma), '_update_frequency_', str(args.update_frequency),
                             '_value_update_times_', str(args.value_update_times)]) + '.png')

    if not args.cuda:
        plt.ioff()

    env.close()

    # Uncomment the last line to report the results to OpenAI if you want.
    # Don't forget to fill in your api_key.
    # gym.upload(results_directory, api_key='')

discounting = [1, 0.9, 0.3, 0.1]
frequency = [5, 10, 15, 20]
times_update = [1, 3, 10]

for discount in discounting:
    for freq in frequency:
        for times in times_update:
            for _ in range(3):
                args_.gamma = discount
                args_.update_frequency = freq
                args_.value_update_times = times
                main(args_)
