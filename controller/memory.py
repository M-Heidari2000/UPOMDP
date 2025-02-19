import torch
import numpy as np
from torch.utils.data import Dataset


class ReplayBuffer:
    """
        Replay buffer holds sample trajectories
    """
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
    ):
        self.capacity = capacity

        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index

    def push(
        self,
        observation,
        action,
        reward,
        done,
    ):
        """
            Add experience (single step) to the replay buffer
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0

    def sample(
        self,
        batch_size: int,
        chunk_length: int
    ):
        done = self.done.copy()
        done[-1] = 1
        episode_ends = np.where(done)[0]

        all_indexes = np.arange(len(self))
        distances = episode_ends[np.searchsorted(episode_ends, all_indexes)] - all_indexes + 1
        valid_indexes = all_indexes[distances >= chunk_length]

        sampled_indexes = np.random.choice(valid_indexes, size=batch_size)
        sampled_ranges = np.vstack([
            np.arange(start, start + chunk_length) for start in sampled_indexes
        ])

        sampled_observations = self.observations[sampled_ranges].reshape(
            batch_size, chunk_length, self.observations.shape[1]
        )
        sampled_actions = self.actions[sampled_ranges].reshape(
            batch_size, chunk_length, self.actions.shape[1]
        )
        sampled_rewards = self.rewards[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_observations, sampled_actions, sampled_rewards, sampled_done
    

class StaticDataset(Dataset):

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        chunk_length: int,
    ):
        self.observations = torch.as_tensor(replay_buffer.observations)
        self.actions = torch.as_tensor(replay_buffer.actions)
        self.rewards = torch.as_tensor(replay_buffer.rewards)
        self.done = torch.as_tensor(replay_buffer.done).clone()
        self.chunk_length = chunk_length

        # compute valid indexes
        self.done[-1] = 1
        episode_ends = torch.where(self.done)[0]

        all_indexes = torch.arange(len(replay_buffer))
        distances = episode_ends[torch.searchsorted(episode_ends, all_indexes)] - all_indexes + 1
        self.valid_indexes = all_indexes[distances >= self.chunk_length]

    def __len__(self):
        return len(self.valid_indexes)
    
    def __getitem__(self, index):
        valid_index = self.valid_indexes[index]

        observations = self.observations[valid_index: valid_index + self.chunk_length]
        actions = self.actions[valid_index: valid_index + self.chunk_length]
        rewards = self.rewards[valid_index: valid_index + self.chunk_length]
        
        return observations, actions, rewards