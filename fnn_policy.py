import argparse
import os


import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


import gym


from model import FNN, Value
from util import Record, Records
from loss import fnn_policy_loss


# Initialize an ArgumentParser instance for setting the conditions for
# a run from the terminal.
parser = argparse.ArgumentParser(description='Approach Acrobot-v1 with Feedforward /'
                                             'policy.')
# The setting of metavar will be the displayed name for argument actions.

parser.add_argument('--update-frequency', type=int, default=1, metavar='N',
                    help='Update the network parameters every args.update_frequency /'
                         'episodes.')
parser.add_argument('--episodes', type=int, default=10000, metavar='N',
                    help='number of episodes for training')
parser.add_argument('--lr-fnn', type=float, default=1e-3, metavar='N',
                    help='learning rate for optimizing FNN policy')
parser.add_argument('--lr-value', type=float, default=1e-2, metavar='N',
                    help='learning rate for optimizing value network')
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
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()


policy = FNN()
policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr_fnn)
value_network = Value()
value_optimizer = optim.Adam(value_network.parameters(), lr=args.lr_value)
mse_loss = nn.MSELoss()


if args.cuda:
    torch.cuda.manual_seed(args.seed)
    FloatTensor = torch.cuda.FloatTensor
    policy.cuda()
    value_network.cuda()
else:
    torch.manual_seed(args.seed)
    FloatTensor = torch.FloatTensor
    import matplotlib.pyplot as plt
    plt.ion()
Tensor = FloatTensor


env = gym.make('Acrobot-v1')
if args.record:
    index = 1
    # Store results from different runs of the model separately.
    results_directory = ''.join(['/tmp', args.directory_name, '/experiment', str(index)])
    dir_exist = os.path.isdir(results_directory)
    while dir_exist:
        index += 1
        results_directory = ''.join(['/tmp', args.directory_name, '/experiment', str(index)])
        dir_exist = os.path.isdir(results_directory)
    env = gym.wrappers.Monitor(env, results_directory)


# Create a list for storing Records.
records = Records(args.update_frequency)
value_losses = []
policy_losses = []
returns = []


for episode in range(1, args.episodes + 1):
    state_ = env.reset()
    done_ = False

    episode_record = []

    for timestep in range(1, 501):
        if not done_:
            if not args.cuda:
                env.render()
            state_ = torch.from_numpy(state_.reshape(1, -1))
            state = Variable(state_, requires_grad=False).type(Tensor)
            p1, p2, p3 = policy.forward(state)
            selection_ = torch.max(torch.cat([p1, p2, p3]), 0)
            p_action = selection_[0].data[0]
            action = selection_[1].data[0]

            next_state_, reward_, done_, info_ = env.step(action[0])
            value = Tensor([reward_]).view(1, -1)
            episode_record.append(Record(state_, action, value, p_action))

            # Update state values
            for i in range(0, len(episode_record)-1):
                episode_record[i].value.add_(value)

            if done_:
                returns.append(episode_record[0].value[0][0])

            state_ = next_state_

    records.push(episode_record)

    if episode % args.update_frequency == 0:
        # Update the value network first.
        value_optimizer.zero_grad()

        history = records.pull()
        state_history = Variable(torch.cat(history.state),
                                 requires_grad=False).type(Tensor)
        value_history = Variable(torch.cat(history.value),
                                 requires_grad=False).type(Tensor)
        probability_history = Variable(torch.cat(history.probability),
                                       requires_grad=True).type(Tensor)
        value = value_network.forward(state_history)
        value_loss = mse_loss(value, value_history)
        value_losses.append(value_loss.data[0])
        value_loss.backward()
        value_optimizer.step()

        # Now update the policy network.
        policy_optimizer.zero_grad()

        value_estimated = value_network.forward(state_history)
        policy_loss = fnn_policy_loss(value_history, value_estimated, probability_history)
        policy_loss = torch.div(policy_loss, records.size())
        policy_losses.append(policy_loss.data[0])
        policy_loss.backward()
        policy_optimizer.step()

        print('====> Episode: {} value_loss: {:.4f} policy_loss: {:.4f} return: {}'.format(episode,
                                                                                           value_loss.data[0],
                                                                                           policy_loss.data[0],
                                                                                           returns[-1]))
        if not args.cuda:
            plt.clf()
            plt.figure(1)
            plt.subplot(311)
            plt.xlabel('episodes')
            plt.ylabel('cumulative rewards')
            plt.plot(returns)
            plt.subplot(312)
            plt.xlabel('update every ' + str(args.update_frequency) + ' episodes')
            plt.ylabel('value network losses')
            plt.plot(value_losses)
            plt.subplot(313)
            plt.xlabel('update every ' + str(args.update_frequency) + ' episodes')
            plt.ylabel('policy losses')
            plt.plot(policy_losses)
            plt.show()
            plt.savefig('training_result_' + str(args.update_frequency) + '.png')

env.close()
plt.ioff()


# Uncomment the last line to report the results to OpenAI if you want.
# Don't forget to fill in your api_key.
# gym.upload(results_directory, api_key='')
