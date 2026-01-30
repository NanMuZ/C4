import numpy as np
import torch
import collections
import sys
import glob
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

def qlearning_dataset_c4(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Similar to the original qlearning_dataset, but returns richer fields and trajectory statistics.

    Returns:
        traj_length: [len(traj_0), len(traj_1), ...]  Number of transitions per trajectory (Note: counts sampled transitions, if terminate_on_end=False, the last step is skipped)
        episode_end_list: [i_0, i_1, ...]             Termination index i in the original dataset for each trajectory (corresponding to the "current step" of the i-th transition)
        data: {
            "start": np.array(start_),                # Whether this is the first transition of a trajectory
            "observations": np.array(obs_),
            "actions": np.array(action_),
            "next_observations": np.array(next_obs_),
            "next_actions": np.array(next_act_),      # Next action a_{t+1}
            "rewards": np.array(reward_),
            "terminals": np.array(done_),
            "timeouts": np.array(timeout_),           # Whether the current transition ended due to timeout
            "trajectory": np.array(traj_),            # Trajectory number (starting from 0)
            "step": np.array(step_),                  # Index i in the original dataset (aligned with obs[i])
            # Optional:
            # "qvel": np.array(qvel_)                 # If dataset contains 'qvel', aligned with current step
            # "qpos": np.array(qpos_)                 # If dataset contains 'qpos', aligned with current step
        }
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    use_timeouts = 'timeouts' in dataset

    # Output buffers
    start_ = []
    obs_ = []
    next_obs_ = []
    action_ = []
    next_act_ = []
    reward_ = []
    done_ = []
    timeout_ = []
    traj_ = []
    step_ = []

    qvel_ = []
    qpos_ = []
    has_qvel = 'qvel' in dataset
    has_qpos = 'qpos' in dataset

    # Trajectory statistics
    traj_length = []          # Final number of transitions per trajectory
    episode_end_list = []     # Original termination index i for each trajectory

    episode_step = 0          # Step count within current trajectory
    curr_traj_len = 0         # Number of transitions collected in current trajectory
    traj_idx = -1             # Trajectory index (starting from -1, set to 0 when entering first trajectory)

    # Iterate to N-2 because we need i+1 as next
    for i in range(N - 1):
        # Read current/next step basic fields
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        act = dataset['actions'][i].astype(np.float32)
        next_act = dataset['actions'][i + 1].astype(np.float32)
        rew = dataset['rewards'][i].astype(np.float32)

        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = bool(dataset['timeouts'][i])
        else:
            # fallback: infer using env._max_episode_steps
            final_timestep = (episode_step == env._max_episode_steps - 1)

        # If this is the first transition of a trajectory, start new trajectory index
        if episode_step == 0:
            traj_idx += 1
            start_flag = True
        else:
            start_flag = False

        # When terminate_on_end=False and final step (timeout or natural end):
        # Skip this transition, but still need to properly end current trajectory and record statistics
        if (not terminate_on_end) and final_timestep:
            # This transition is not written to data, but trajectory terminates at i
            episode_end_list.append(i)
            traj_length.append(curr_traj_len)  # Number of data entries already written
            # Reset and start new trajectory
            episode_step = 0
            curr_traj_len = 0
            continue

        # Normal write of this transition
        start_.append(start_flag)
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(act)
        next_act_.append(next_act)
        reward_.append(rew)
        done_.append(done_bool)
        timeout_.append(final_timestep)
        traj_.append(traj_idx)
        step_.append(i)

        if has_qvel:
            qvel_.append(np.array(dataset['qvel'][i], dtype=np.float32))
        if has_qpos:
            qpos_.append(np.array(dataset['qpos'][i], dtype=np.float32))

        episode_step += 1
        curr_traj_len += 1

        # If trajectory ends at this step (terminal or timeout), finalize and reset
        if done_bool or final_timestep:
            episode_end_list.append(i)
            traj_length.append(curr_traj_len)
            episode_step = 0
            curr_traj_len = 0

    data = {
        "start": np.array(start_, dtype=bool),
        "observations": np.array(obs_, dtype=np.float32),
        "actions": np.array(action_, dtype=np.float32),
        "next_observations": np.array(next_obs_, dtype=np.float32),
        "next_actions": np.array(next_act_, dtype=np.float32),
        "rewards": np.array(reward_, dtype=np.float32),
        "terminals": np.array(done_, dtype=bool),
        "timeouts": np.array(timeout_, dtype=bool),
        "trajectory": np.array(traj_, dtype=np.int64),
        "step": np.array(step_, dtype=np.int64),
    }
    if has_qvel:
        data["qvel"] = np.array(qvel_, dtype=np.float32)
    if has_qpos:
        data["qpos"] = np.array(qpos_, dtype=np.float32)

    return traj_length, episode_end_list, data

