import errno
import os
import time

import gym
import torch as th
from torch.autograd import Variable

from loss import fnn_policy_loss
from util import Records, Record


def fnn_policy_train(env,
                     policy,
                     value_network,
                     policy_optimizer,
                     value_optimizer,
                     value_loss_function,
                     args,
                     convergence_threshold=0.1):
    """

    Parameters
    ----------
    env: gym.wrappers.time_limit.TimeLimit
        unwrapped gym simulated environment
    policy: nn.module
        model that learns a categorical distribution for p(a|s)
    value_network: nn.module
        model that learns state value function V(s)
    policy_optimizer: torch.optim.Optimizer
        optimizer that optimizes the parameters of the policy network
    value_optimizer: torch.optim.Optimizer
        optimizer that optimizes the parameters of the value network
    value_loss_function: torch.nn.modules.loss
        loss function for the value network
    args: argparse.Namespace
        args that specify the setting for an experiment
    convergence_threshold: float
        threshold that decides if the convergence of the policy is attained

    Returns
    -------
    time_str: str
        string of the format YearMonthDay-HourMinuteSeconds, which is used in naming files for distinguishing
        between different experiments
    policy_trained: nn.module
        policy model that has been trained
    """
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if args.record:
        # Store results from different runs of the model separately.
        results_directory = ''.join(['/tmp', args.directory_name, '/training/', time_str, '_discounting_',
                                     str(args.gamma), '_update_frequency_', str(args.update_frequency),
                                     '_value_update_times_', str(args.value_update_times)])
        env = gym.wrappers.Monitor(env, results_directory)

    if args.cuda:
        th.cuda.manual_seed(args.seed)
        FloatTensor = th.cuda.FloatTensor
        policy.cuda()
        value_network.cuda()
    else:
        th.manual_seed(args.seed)
        FloatTensor = th.FloatTensor
        import matplotlib.pyplot as plt
        plt.ion()
        try:
            os.makedirs('fnn/training')
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    Tensor = FloatTensor

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
                state_ = th.from_numpy(state_.reshape(1, -1))
                state = Variable(state_, requires_grad=False).type(Tensor)
                p = policy.forward(state)
                selection_ = th.multinomial(p, 1)
                action = selection_.data[0]
                p_action = p[0][action].view(1, -1)

                next_state_, reward_, done_, info_ = env.step(action[0])
                value = Tensor([reward_]).view(1, -1)
                episode_record.append(Record(state_, action, value, p_action, p.view(1, -1)))

                # Update state values
                for i in range(0, len(episode_record)-1):
                    episode_record[i].value.add_((args.gamma ** (len(episode_record)-1-i)) * value)

                if done_:
                    returns.append(episode_record[0].value[0][0])

                state_ = next_state_

        records.push(episode_record)

        if episode % args.update_frequency == 0:

            history = records.pull()
            state_history = Variable(th.cat(history.state),
                                     requires_grad=False).type(Tensor)
            value_history = Variable(th.cat(history.value),
                                     requires_grad=False).type(Tensor)
            prob_action_history = th.cat(history.probability_action).type(Tensor)
            prob_history = th.cat(history.probabilities).type(Tensor)

            # Update the value network first.
            for _ in range(args.value_update_times):
                value_optimizer.zero_grad()
                value = value_network.forward(state_history)
                value_loss = value_loss_function(value, value_history)
                value_losses.append(value_loss.data[0])
                value_loss.backward()
                value_optimizer.step()

            # Now update the policy network.
            policy_optimizer.zero_grad()

            value_estimated_ = value_network.forward(state_history)
            value_estimated = value_estimated_.detach()
            policy_loss = fnn_policy_loss(value_history, value_estimated, prob_action_history.view(-1, 1),
                                          prob_history.view(-1, 3))
            policy_loss = th.div(policy_loss, state_history.size()[0])
            policy_losses.append(policy_loss.data[0])

            if len(policy_losses) >= 2:
                if abs(policy_losses[-1] - policy_losses[-2]) < convergence_threshold:
                    if not args.cuda:
                        plt.ioff()
                        plt.close()
                    env.close()
                    return time_str, policy

            policy_loss.backward()

            policy_optimizer.step()

            print('====> Episode: {} value_loss: {:.4f} policy_loss: {:.4f} return: {}'.format(episode,
                                                                                               value_losses[-1],
                                                                                               policy_losses[-1],
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
                plt.savefig(''.join(['fnn/training/', time_str, '_discounting_',
                                     str(args.gamma), '_update_frequency_', str(args.update_frequency),
                                     '_value_update_times_', str(args.value_update_times)]) + '.png')

    if not args.cuda:
        plt.ioff()
        plt.close()

    env.close()

    return time_str, policy
