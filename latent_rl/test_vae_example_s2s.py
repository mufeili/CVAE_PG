import argparse
import gym
import numpy as np
import random
import visdom
from model import Policy, VAE

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Test VAE trained')
parser.add_argument('--policy-dir', type=str, default='actor_critic_20171020-082409',
                    help='directory for trained policy to generate experience')
parser.add_argument('--VAE-dir', type=str, default='VAE_size512_batch_False20171020-164149',
                    help='directory for trained VAE model')

args = parser.parse_args()

env1 = gym.make('Acrobot-v1')
env2 = gym.make('Acrobot-v1')

policy = torch.load(args.policy_dir)
vae = torch.load(args.VAE_dir)


def select_action(state_):
    state = torch.from_numpy(state_).float().unsqueeze(0)
    probs, _ = policy(Variable(state))
    action_ = probs.multinomial()
    return action_.data

state_ = env1.reset()
env2.reset()
done = False

states = [env1.env.state]

while not done:
    action = select_action(state_)
    state_, reward, done, info = env1.step(action[0, 0])
    states.append(env1.env.state)

sample_batch = random.sample(states, 6)
vis = visdom.Visdom()

for i in range(len(sample_batch)):
    sample_ = sample_batch[i]
    env1.env.state = sample_
    image1 = env1.render(mode='rgb_array')
    vis.image(np.transpose(image1, (2, 0, 1)), opts=dict(title=''.join(['Ground truth', str(i)])))
    sample = Variable(torch.FloatTensor([sample_]), requires_grad=False)
    recon_state, _, _ = vae(sample)
    env2.env.state = recon_state.data.numpy().reshape(4,)
    image2 = env2.render(mode='rgb_array')
    vis.image(np.transpose(image2, (2, 0, 1)), opts=dict(title=''.join(['Reconstructed', str(i)])))

env1.close()
env2.close()
