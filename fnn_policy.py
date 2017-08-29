import argparse
import errno
import os


import torch as th
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


import gym


from model import Fnn, Value
from fnn_policy_train import fnn_policy_train


# Initialize an ArgumentParser instance for setting the conditions for
# a run from the terminal.
parser = argparse.ArgumentParser(description='Approach Acrobot-v1 with Feedforward /'
                                             'policy.')
# The setting of metavar will be the displayed name for argument actions.

parser.add_argument('--update-frequency', type=int, default=5, metavar='N',
                    help='Update the network parameters every args.update_frequency /'
                         'episodes.')
parser.add_argument('--value-update-times', type=int, default=1, metavar='N',
                    help='Times to update a value network when it is the time for update')
parser.add_argument('--episodes', type=int, default=10000, metavar='N',
                    help='number of episodes for training')
parser.add_argument('--lr-fnn', type=float, default=1e-3, metavar='N',
                    help='learning rate for optimizing FNN policy')
parser.add_argument('--lr-value', type=float, default=1e-2, metavar='N',
                    help='learning rate for optimizing value network')
parser.add_argument('--gamma', type=float, default=1, metavar='N',
                    help='discounting factor for the cumulative return')
parser.add_argument('--test-time', type=float, default=100, metavar='N',
                    help='number of times for test')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# If the option is specified, set args.no_cuda = True, otherwise it
# is False by default.
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--directory-name', type=str, default='/acrobot-results/fnn',
                    metavar='D', help='directory for storing results (default: '
                                      '/acrobot-results/fnn)')
parser.add_argument('--record', action='store_true', default=False,
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
    try:
        os.makedirs('fnn/test')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
Tensor = FloatTensor


def main(args):
    policy = Fnn()
    if args.cuda:
        policy.cuda()
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr_fnn)
    value_network = Value()
    if args.cuda:
        value_network.cuda()
    value_optimizer = optim.Adam(value_network.parameters(), lr=args.lr_value)
    mse_loss = nn.MSELoss()

    env = gym.make('Acrobot-v1')

    time_str, trained_model = fnn_policy_train(env,
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

    test_returns = []

    if not args.cuda:
        plt.ion()

    for i in range(args.test_time):
        state_ = env.reset()
        done = False
        cumulative_return = 0

        for timestep in range(0, 500):
            if not done:
                if not args.cuda:
                    env.render()
                state_ = th.from_numpy(state_.reshape(1, -1))
                state = Variable(state_, requires_grad=False).type(Tensor)
                p = trained_model.forward(state)
                action = th.max(p, 1)[1].data[0]
                next_state_, reward_, done, info_ = env.step(action)
                cumulative_return += (args.gamma ** timestep) * reward_
                state_ = next_state_

        test_returns.append(cumulative_return)

        print('====> Cumulative return: {}'.format(cumulative_return))

        if not args.cuda:
            plt.clf()
            plt.figure(1)
            plt.xlabel('episodes')
            plt.ylabel('cumulative returns')
            plt.plot(list(test_returns))
            plt.show()
            plt.savefig(''.join(['fnn/test/', time_str, '_discounting_',
                        str(args.gamma), '_update_frequency_', str(args.update_frequency),
                        '_value_update_times_', str(args.value_update_times)]) + '.png')

    if not args.cuda:
        plt.ioff()
        plt.close()

    env.close()

    # Uncomment the last line to report the results to OpenAI if you want.
    # Don't forget to fill in your api_key.
    # gym.upload(results_directory, api_key='')

    return sum(test_returns)/len(test_returns)

max_averaged_returns = -500
max_model_parameters = (0, 0, 0)

hyperparameters = [(1, 20, 1), (1, 5, 3), (1, 10, 3), (1, 20, 3), (1, 5, 10),
                   (1, 10, 10), (1, 15, 10)]

for discount, freq, times in hyperparameters:
    for _ in range(3):
        args_.gamma = discount
        args_.update_frequency = freq
        args_.value_update_times = times

        max_averaged_returns_ = max_averaged_returns
        averaged_return = main(args_)
        if averaged_return > max_averaged_returns_:
            max_averaged_returns_ = averaged_return
        if max_averaged_returns != max_averaged_returns_:
            print('====> discount: {}, update frequency: {}, update times: {}, '
                  'max averaged return: {}'.format(discount, freq, times,
                                                   max_averaged_returns_))
        max_averaged_returns = max_averaged_returns_
