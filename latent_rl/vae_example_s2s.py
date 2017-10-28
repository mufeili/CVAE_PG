from __future__ import print_function
import argparse
import gym
import time
from model import Policy, VAE
from util import Logger, ReplayBuffer, Transition

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='reconstruct state from state using VAE')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--iterations', type=int, default=10000, metavar='N',
                    help='number of iterations to train for each epoch(default: 10000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train(default: 10)')
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--buffer-capacity', type=int, default=50000, metavar='N',
                    help='capacity for the replay buffer')
parser.add_argument('--loss', type=str, default='l2', help='loss function to use')
parser.add_argument('--batch-norm', action='store_true', default=False,
                    help='whether to use batch normalization in VAE')
parser.add_argument('--kl-divergence', action='store_true', default=False,
                    help='use kl divergence in the loss function or not')
parser.add_argument('--policy-dir', type=str, default='actor_critic_20171020-082409',
                    help='directory for trained policy to generate experience')
parser.add_argument('--kl-weight', type=float, default=1,
                    help='hyperparameter that discounts kl divergence')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


model = VAE(input_size=4, hidden1_size=8, hidden2_size=4,
            batch_norm=args.batch_norm)
if args.cuda:
    model.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

if args.loss == 'l1':
    reconstruction_function = nn.L1Loss()
else:
    reconstruction_function = nn.MSELoss()


def loss_function(recon_x, x, mu, logvar, logger, timestep):
    recon_loss = reconstruction_function(recon_x, x)
    if args.loss == 'l1':
        logger.scalar_summary('l1 loss', recon_loss.data[0], timestep)
    else:
        logger.scalar_summary('MSE loss', recon_loss.data[0], timestep)

    # See Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    logger.scalar_summary('KL divergence', KLD.data[0], timestep)

    if args.kl_divergence:
        return recon_loss + args.kl_weight * KLD
    else:
        return recon_loss


optimizer = optim.Adam(model.parameters(), lr=5e-4)

policy = torch.load(args.policy_dir)

env = gym.make('Acrobot-v1')
env.seed(args.seed)
# Set the logger.
if args.kl_divergence:
    logger = Logger('./logs_vae')
else:
    logger = Logger('./logs_vae_nokl')


def select_action(state_):
    state = torch.from_numpy(state_).float().unsqueeze(0)
    probs, _ = policy(Variable(state))
    action_ = probs.multinomial()
    return action_.data

model.train()

for epoch in range(args.epochs):
    buffer = ReplayBuffer(args.buffer_capacity)
    while len(buffer) < args.buffer_capacity:
        state = env.reset()
        # Note in state to state we need states only.
        buffer.push(Tensor([env.env.state]), None, None, None, None)
        done = False
        while not done:
            action = select_action(state)
            state, reward, done, _ = env.step(action[0, 0])
            # Note in state to state we need states only.
            buffer.push(Tensor([env.env.state]), None, None, None, None)

    for i in range(args.iterations):
        sample = buffer.sample(args.batch_size)
        batch = Transition(*zip(*sample))

        state_batch = Variable(torch.cat(batch.state), requires_grad=False).type(Tensor)
        recon_batch, mu, logvar = model(state_batch)
        loss = loss_function(recon_batch, state_batch, mu, logvar, logger, epoch * args.iterations + i + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Epoch: {} Iteration: {} Loss:{:.6f}'.format(
                epoch + 1, i + 1, loss.data[0]
            ))

env.close()
model.eval()

time_str = time.strftime("%Y%m%d-%H%M%S")
torch.save(model, ''.join(['VAE_size', str(args.batch_size), '_batch_', str(args.batch_norm), time_str]))
