from collections import namedtuple
from io import BytesIO
import numpy as np
import random
import scipy.misc
import tensorflow as tf
import torch
import torch.nn as nn


def vae_loss_function(recon_x, x, mu, log_var, logger, timestep,
             use_kl=True, **kwargs):
    if use_kl:
        kl_discount = kwargs.get('kl_discount', 0)

    recon_loss = nn.MSELoss()(recon_x, x)
    logger.scalar_summary('MSE loss', recon_loss.data[0], timestep)

    # See Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence_ = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    kl_divergence = torch.sum(kl_divergence_).mul_(-0.5)
    logger.scalar_summary('KL divergence', kl_divergence.data[0], timestep)

    if use_kl:
        return recon_loss + kl_discount * kl_divergence
    else:
        return recon_loss


# Create a new tuple subclass called Transition.
Transition = namedtuple('Transition', 'state')


class ReplayBuffer(object):
    """The file implements the replay_buffer for experience replay in DQN.
    The code is mainly based on http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""
    def __init__(self, capacity):
        """
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
        """
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def push(self, *args):
        """
        Saves a transition.
        Parameters
        ----------
        *args:
        - state: torch.FloatTensor
            state/observation before transition
        - action: torch.LongTensor
            action to take at the state
        - next_state: torch.FloatTensor
            state/observation after transition
        - reward: torch.FloatTensor
            signal representing the immediate feedback about
            how good it is to be in next_state
        """
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        """
        Sample #batch_size experiences without replacement.
        Parameters
        ----------
        batch_size: int
            number of Transitions to sample
        Returns
        -------
        sample_result: list
            a list of sampled Transitions
        """
        sample_result = random.sample(self._memory, batch_size)
        return sample_result

    def __len__(self):
        return len(self._memory)


class Logger(object):
    """Code from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py"""
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
