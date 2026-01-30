import random
import numpy as np
from offlinerlkit.buffer import ReplayBuffer


def sample(buffer: ReplayBuffer, data_size: int, local_data_size: int):
    """
    Randomly sample specified amount of data from buffer
    
    Args:
        buffer: ReplayBuffer object
        data_size: total data size
        local_data_size: data size to sample
    
    Returns:
        sampled data
    """
    length = data_size
    assert 1 <= local_data_size
    
    indices = np.random.randint(0, length, local_data_size)
    return buffer[indices]


import random

def collect_replaytrajs(start_list: list, end_list: list, local_data_size: int, traj_seed: int = 0, full: bool = False):
    """
    Collect replay trajectories, suitable for replay and antmaze tasks.
    - full=False: Randomly sample from trajectories, longer trajectories have higher probability of being selected, until sampled points reach local_data_size.
    - full=True: Directly return all complete trajectories.

    Args:
        start_list: list of trajectory start positions
        end_list: list of trajectory end positions
        local_data_size: local data size (only used when full=False)
        traj_seed: random seed
        full: whether to directly return complete trajectories (default False)

    Returns:
        collected_trajs: list of sampled trajectory segments ([(start, end), ...])
        total_length_list: length of each trajectory segment
    """
    # Generate number of points for each trajectory
    traj_lengths = [end - start + 1 for start, end in zip(start_list, end_list)]

    if full:
        # Directly return all complete trajectories
        collected_trajs = [(start, end) for start, end in zip(start_list, end_list)]
        total_length_list = traj_lengths
        return collected_trajs, total_length_list

    # Random seed initialization
    random.seed(traj_seed)

    # Variables for collecting results
    collected_trajs = []
    total_length_collected = 0
    total_length_list = []

    # Sample from trajectories until reaching local_data_size
    while total_length_collected < local_data_size:
        # Randomly select a trajectory based on length proportion
        selected_traj_index = random.choices(range(len(start_list)), weights=traj_lengths, k=1)[0]
        start, end = start_list[selected_traj_index], end_list[selected_traj_index]
        traj_length = traj_lengths[selected_traj_index]

        # Calculate remaining length that can be sampled from this trajectory
        remaining_length = local_data_size - total_length_collected

        # Determine whether to sample the entire trajectory or just part of it
        if remaining_length >= traj_length:
            collected_trajs.append((start, end))
            total_length_list.append(traj_length)
            total_length_collected += traj_length
        else:
            collected_trajs.append((start, start + remaining_length - 1))
            total_length_list.append(remaining_length)
            total_length_collected += remaining_length

    return collected_trajs, total_length_list





def collect_trajs(start_list: int, end_list: int, local_data_size: int, traj_seed: int=0):
    """
    Collect trajectory data, shuffle randomly and add as many trajectories as possible
    
    Args:
        start_list: list of trajectory start positions
        end_list: list of trajectory end positions
        local_data_size: local data size
    
    Returns:
        list of collected trajectory tuples and length list
    """
    # Calculate length of each trajectory
    traj_lengths = [end - start + 1 for start, end in zip(start_list, end_list)]
    
    # Combine start positions and trajectory lengths for easier processing
    trajs = list(zip(start_list, traj_lengths))
    
    # Randomly shuffle trajectories
    random.seed(traj_seed)
    random.shuffle(trajs)
    
    # Variables for collecting results
    collected_trajs = []
    total_length = 0
    total_length_list = []
    
    # Iterate through trajectories, add as many as possible
    for start, length in trajs:
        if length > 0:
            if total_length + length > local_data_size:
                # Check if the last trajectory needs to be truncated
                remaining_length = local_data_size - total_length
                if remaining_length > 0:
                    collected_trajs.append((start, start + remaining_length - 1))
                    total_length_list.append(remaining_length)
                    total_length += remaining_length
                break
            collected_trajs.append((start, start + length - 1))
            total_length_list.append(length)
            total_length += length

    return collected_trajs, total_length_list


def extract_and_combine_trajs(dataset, collected_trajs):
    """
    Extract and combine specified trajectory segments from dataset
    
    Args:
        dataset: dictionary containing observations, actions, etc.
        collected_trajs: list of collected trajectory start and end position tuples
    
    Returns:
        combined new dataset dictionary
    """
    # Create a new dictionary to store extracted data
    new_dataset = {key: [] for key in dataset.keys()}
    
    # Iterate through all collected trajectory intervals
    for start, end in collected_trajs:
        for key in dataset.keys():
            # Extract corresponding trajectory segment from each key's array
            segment = dataset[key][start:end+1]  # +1 because end is inclusive
            # Add segment to new list
            new_dataset[key].append(segment)
    
    # Use np.concatenate to concatenate all arrays in the list along the first axis
    for key in new_dataset.keys():
        new_dataset[key] = np.concatenate(new_dataset[key], axis=0)
    
    return new_dataset
