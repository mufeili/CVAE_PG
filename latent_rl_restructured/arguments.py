import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='a|z(s, s_next)',
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
    parser.add_argument('--buffer-capacity', type=int, default=5000, metavar='N',
                        help='capacity for the replay buffer')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--vae-update-frequency', type=int, default=10, metavar='N',
                        help='update vae and empty the replay buffer every vae_update_frequency times')
    parser.add_argument('--vae-update-times', type=int, default=10, metavar='N',
                        help='update vae for vae_update_times every update')
    parser.add_argument('--vae-update-threshold', type=float, default=0.02, metavar='N',
                        help='check if the vae loss no longer decreases between checks')
    parser.add_argument('--vae-check-interval', type=int, default=500, metavar='N',
                        help='interval between two checks for vae loss')
    parser.add_argument('--kl-weight', type=float, default=1,
                        help='hyperparameter that discounts kl divergence')
    parser.add_argument('--bp2VAE', action='store_true', default=False,
                        help='whether back propagate the gradient of the policy to VAE')
    args = parser.parse_args()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args
