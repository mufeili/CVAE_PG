"""The file implements the replay_buffer for experience replay in DQN.
The code is mainly based on http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""
from collections import namedtuple
import random


# Create a new tuple subclass called Transition.
Transition = namedtuple('Transition',
                        ('state', 'action', 'cum_return'))


class ReplayBuffer(object):

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
