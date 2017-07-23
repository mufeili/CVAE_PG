from collections import namedtuple


# Create a new tuple subclass called Record.
Record = namedtuple('Record',
                    ('state', 'action', 'value', 'probability'))


class Records:
    """
    Store Record from a number of episodes and keep them separately.
    """
    def __init__(self, capacity):
        """

        Parameters
        ----------
        capacity: int
            Max number of episodes to store. When the Records overflow
            the old memories are dropped.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode_record):
        """
        Save an episode of experiences.

        Parameters
        ----------
        episode_record: list
            a list of Record for the episode
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode_record
        self.position = (self.position + 1) % self.capacity

    def pull(self):
        """
        Return records in a form that is convenient for training.

        Returns
        -------
        history: Record object
            It consists of experiences from all episodes.
        """
        all_episodes = sum(self.memory, [])
        history = Record(*zip(*all_episodes))
        return history

    def __len__(self):
        return len(self.memory)

    def size(self):
        """

        Returns
        -------
        size: int
            the total number of experiences stored
        """
        size = 0
        for i in range(len(self.memory)):
            size += len(self.memory[i])
        return size
