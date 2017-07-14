"""
We referred to the examples provided by
https://github.com/pytorch/examples/tree/master/vae and
https://github.com/wiseodd/generative-models/tree/master/VAE/conditional_vae
"""
import argparse
from collections import namedtuple


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


import gym


# Initialize an ArgumentParser instance for setting the conditions for
# a run from the terminal.
parser = argparse.ArgumentParser(description='Approach Acrobot-v1 with Policy\ '
                                             'Gradient using conditional\ '
                                             'Variational Autoencoder')
# The setting of metavar will be the displayed name for argument actions.

# The number of timesteps to run for each epoch is set to be 500 because the
# Acrobot-v1 task is set to end in 500 time steps in gym.
parser.add_argument('--timesteps', type=int, default=500, metavar='N',
                    help='number of timesteps for collecting data in each epoch')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs for training')
# If the option is specified, set args.no_cuda = True, otherwise it
# is False by default.
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--directory-name', type=str, default='/acrobot-results',
                    metavar='D', help='directory for storing results (default: '
                                      '/acrobot-results)')
parser.add_argument('--record', action='store_true', default=False,
                    help='enables recording gym results')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()


if args.cuda:
    torch.cuda.manual_seed(args.seed)
    FloatTensor = torch.cuda.FloatTensor
else:
    torch.manual_seed(args.seed)
    FloatTensor = torch.FloatTensor
    import matplotlib.pyplot as plt
    plt.ion()
Tensor = FloatTensor


class VAE(nn.Module):
    """
    Implementation of VAE model. Strictly speaking, it is more like a
    variation of conditional VAE.
    """
    def __init__(self):
        super(VAE, self).__init__()
        # behavior policy(forward)
        self.fc1 = nn.Linear(6, 12)
        self.fc21 = nn.Linear(12, 24)
        self.fc22 = nn.Linear(12, 24)
        self.fc3 = nn.Linear(30, 3)

        # target policy(reconstruct)
        self.fc41 = nn.Linear(9, 24)
        self.fc42 = nn.Linear(9, 24)
        self.fc5 = nn.Linear(30, 3)

    def encode(self, s):
        """
        Encode the state and get the intermediate params for h.

        Parameters
        ----------
        s: Variable whose data is Tensor of shape n x 6 with requires_grad=False
            state signal sent by the environment

        Returns
        -------
        mu_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            mean of the modeled gaussian distribution for h
        log_var_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            logarithm of the covariance of the modeled gaussian distribution for h
        """
        h_ = self.fc1(s)
        return self.fc21(h_), self.fc22(h_)

    @staticmethod
    def reparameterize(mu_h, log_var_h):
        """
        Perform the reparameterization trick. The reparameterization trick in VAE
        makes it tractable for BP to deal with the distribution of h.

        Parameters
        ----------
        mu_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            mean of the modeled gaussian distribution for h
        log_var_h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            logarithm of the covariance of the modeled gaussian distribution for h

        Returns
        -------
        h_sampled: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            h_sampled = mu_h + epsilon x covariance, where epsilon is sampled from a
            standard normal distribution
        """
        std = log_var_h.mul(0.5).exp_()
        epsilon = Tensor(std.size()).normal_()
        epsilon = Variable(epsilon, requires_grad=False).type(Tensor)
        return epsilon.mul(std).add_(mu_h)

    def select_action(self, h, s):
        """
        Return scores separately for three discrete actions, argmax will be
        performed based on that.

        Parameters
        ----------
        h: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            sampled latent variable
        s: Variable whose data is Tensor of shape n x 6 with requires_grad=False
            state signal sent by the environment

        Returns
        -------
        a: Variable whose data is Tensor of shape n x 3 with requires_grad=True
            scores for the three discrete actions
        """
        h_s = torch.cat([h, s], 1)
        return self.fc3(h_s)

    def forward(self, s):
        """
        See self.encode, self.reparameterize and self.select_action
        """
        mu_f, log_var_f = self.encode(s)
        h_sampled = self.reparameterize(mu_f, log_var_f)
        a = self.select_action(h_sampled, s)
        return mu_f, log_var_f, a

    def reconstruct(self, s, a):
        """
        Reconstruct the distribution for h and action based on the state and the
        true action.

        Parameters
        ----------
        s: Variable whose data is Tensor of shape n x 6 with requires_grad=False
            state signal sent by the environment
        a: Variable whose data is Tensor of shape n x 3 with requires_grad=True
            scores for the three discrete actions

        Returns
        -------
        mu_reconstruct: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            reconstructed mean for the modeled Gaussian distribution of h
        log_var_reconstruct: Variable whose data is Tensor of shape n x 24 with requires_grad=True
            reconstructed logarithm of the covariance for the modeled Gaussian distribution of h
        a_reconstruct: Variable whose data is Tensor of shape n x 3 with requires_grad=True
            reconstructed scores for the three discrete actions
        """
        s_a = torch.cat([s, a], 1)
        mu_reconstruct = self.fc41(s_a)
        log_var_reconstruct = self.fc42(s_a)
        h_reconstruct = self.reparameterize(mu_reconstruct, log_var_reconstruct)
        h_s_reconstruct = torch.cat([h_reconstruct, s], 1)
        return mu_reconstruct, log_var_reconstruct, self.fc5(h_s_reconstruct)


model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


if args.cuda:
    model.cuda()


def mse_loss(inputs, targets):
    """
    The nn criterion does not compute the gradient w.r.t. targets, so we need to do
    it ourselves.
    """
    return torch.sum((inputs - targets) ** 2)


def loss_function(q_value,
                  mu_true,
                  log_var_true,
                  mu_reconstruct,
                  log_var_reconstruct,
                  action_true,
                  action_reconstruct):
    """
    loss function -Q(s,a)*(square_loss+KL(q|p)), where q is the reconstructed distribution
    for h and p is the original distribution for h.

    Parameters
    ----------
    q_value: Variable whose data is Tensor of shape 1 with requires_grad=False
        the un-discounted cumulative reward for the epoch
    mu_true: Variable whose data is Tensor of shape n x 24 with requires_grad=True
        the true mean for the modeled distribution of h
    log_var_true: Variable whose data is Tensor of shape n x 24 with requires_grad=True
        the true logarithm of the covariance for the modeled distribution of h
    mu_reconstruct: Variable whose data is Tensor of shape n x 24 with requires_grad=True
        the reconstructed mean for the modeled distribution of h
    log_var_reconstruct: Variable whose data is Tensor of shape n x 24 with requires_grad=True
        the reconstructed logarithm of the covariance for the modeled distribution of h
    action_true: Variable whose data is Tensor of shape n x 3 with requires_grad=True
        the true scores for the three discrete actions
    action_reconstruct: Variable whose data is Tensor of shape n x 3 with requires_grad=True
        the reconstructed scores for the three discrete actions

    Returns
    -------
    loss: Variable whose data is Tensor of shape 1, with requires_grad=True
        the loss calculated
    """

    # l2 loss
    l2_loss = mse_loss(action_true, action_reconstruct)

    # The KLE divergence between two multivariate Gaussian N(\mu_{1}, \Sigma_{1})
    # and N(\mu_{0}, \Sigma_{0}) is 0.5 * (tr(\Sigma_{0}^{-1}\Sigma_{1})+
    # (\mu_{0}-\mu_{1})^{T}\Sigma_{0}^{-1}(\mu_{0}-\mu_{1})-k+
    # \log(\det\Sigma_{0}/\det\Sigma_{1})
    var = log_var_true.exp()
    var_inverse = Variable(torch.ones(var.size())/var.data, requires_grad=True).type(Tensor)
    kl_divergence = 0.5 * (var_inverse.dot(log_var_reconstruct.exp()) +
                           torch.sum((mu_true-mu_reconstruct).mul(var_inverse).
                                     mul(mu_true-mu_reconstruct)) - var.size()[1] * var.size()[0] +
                           torch.sum(log_var_true-log_var_reconstruct))
    return - q_value * (l2_loss + kl_divergence) / var.size()[0]


# Create a new tuple subclass called Record.
Record = namedtuple('Record',
                    ('state', 'action', 'reward', 'mu', 'log_var'))
# Create a list for storing Records.
records = []


# Create a list for storing returns.
returns = []
# Create a list for storing losses.
losses = []


env = gym.make('Acrobot-v1')
if args.record:
    results_directory = ''.join(['/tmp', args.directory_name])
    env = gym.wrappers.Monitor(env, results_directory)


for epoch in range(1, args.epochs + 1):
    state_ = env.reset()

    # policy run
    for timesteps in range(1, args.timesteps + 1):
        if not args.cuda:
            env.render()
        state_ = torch.from_numpy(state_.reshape(1, -1))
        state = Variable(state_, requires_grad=False).type(Tensor)
        mu, log_var, action_ = model.forward(state)
        action = torch.max(action_, 1)[1][0].data[0]
        next_state, reward_, done_, info_ = env.step(action)
        reward = Tensor([reward_]).view(1, -1)
        records.append(Record(state_, action_.data.view(1, -1), reward,
                              mu.data.view(1, -1), log_var.data.view(1, -1)))
        state_ = next_state

    # reconstruct
    history = Record(*zip(*records))
    state_history = Variable(torch.cat(history.state), requires_grad=False).type(Tensor)
    action_history = Variable(torch.cat(history.action), requires_grad=True).type(Tensor)
    cum_reward_ = torch.sum(torch.cat(history.reward))
    returns.append(cum_reward_)
    cum_reward = Variable(Tensor([cum_reward_]), requires_grad=False).type(Tensor)
    mu_history = Variable(torch.cat(history.mu), requires_grad=True).type(Tensor)
    log_var_history = Variable(torch.cat(history.log_var), requires_grad=True).type(Tensor)
    mu_re, log_var_re, action_re = model.reconstruct(state_history, action_history)

    optimizer.zero_grad()

    loss = loss_function(cum_reward, mu_history, log_var_history, mu_re, log_var_re, action_history, action_re)
    losses.append(loss.data[0])
    loss.backward()

    optimizer.step()

    records = []

    print('====> Epoch: {} loss: {:.4f} return: {}'.format(epoch, loss.data[0], cum_reward_))
    if epoch % 20 == 0 and (not args.cuda):
        plt.clf()
        plt.figure(1)
        plt.subplot(211)
        plt.xlabel('epochs')
        plt.ylabel('cumulative rewards')
        plt.plot(returns)
        plt.subplot(212)
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.plot(losses)
        plt.show()
        plt.savefig('training_result.png')


env.close()
plt.ioff()


# Uncomment the last line to report the results to OpenAI if you want.
# Don't forget to fill in your api_key.
# gym.upload(results_directory, api_key='')
