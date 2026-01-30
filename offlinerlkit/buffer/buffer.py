import numpy as np
import torch
import sys
import random
import os
from offlinerlkit.utils.custom_gmm import Cluster4C4
from copy import deepcopy
import time
from datetime import datetime
from typing import Optional, Union, Tuple, Dict, List

# Clustering related imports
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.next_actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.rewards0 = np.zeros((self._max_size, 1), dtype=np.float32)
        self.rewards1 = np.zeros((self._max_size, 1), dtype=np.float32)
        self.rewards2 = np.zeros((self._max_size, 1), dtype=np.float32)
        self.rewards3 = np.zeros((self._max_size, 1), dtype=np.float32)
        self.rewards4 = np.zeros((self._max_size, 1), dtype=np.float32)
        self.fake_rewards = np.zeros((self._max_size, 1), dtype=np.float32)

        self.trajectory = np.zeros((self._max_size, 1), dtype=np.float32)
        self.step = np.zeros((self._max_size, 1), dtype=np.float32)

        self.num_rewards = 0

        self.device = torch.device(device)
        
        # self.num_rewards = num_rewards

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def add_batch_TR(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        print(batch_size)
        print(self._max_size)
        print(indexes[-10:])

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    

        
    def load_dataset_R(self, dataset: Dict[str, np.ndarray], rewards_tmp: np.float32) -> None:
        print("load_dataset1")
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        rewards_tmp_arre = np.array(rewards_tmp, dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards_tmp_arre
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
        
    def load_datasetn(self, dataset: Dict[str, np.ndarray], critics_number: int, rewards1: np.float32, rewards2: np.float32) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        rewards1_arre = np.array(rewards1, dtype=np.float32).reshape(-1, 1)
        rewards2_arre = np.array(rewards2, dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.rewards1 = rewards1_arre
        self.rewards2 = rewards2_arre

        self._ptr = len(observations)
        self._size = len(observations)
        
    def load_dataset5(self, dataset: Dict[str, np.ndarray], critics_number: int, rewards0: np.float32, rewards1: np.float32, rewards2: np.float32, rewards3: np.float32, rewards4: np.float32) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        rewards0_arre = np.array(rewards0, dtype=np.float32).reshape(-1, 1)
        rewards1_arre = np.array(rewards1, dtype=np.float32).reshape(-1, 1)
        rewards2_arre = np.array(rewards2, dtype=np.float32).reshape(-1, 1)
        rewards3_arre = np.array(rewards3, dtype=np.float32).reshape(-1, 1)
        rewards4_arre = np.array(rewards4, dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.rewards0 = rewards0_arre
        self.rewards1 = rewards1_arre
        self.rewards2 = rewards2_arre
        self.rewards3 = rewards3_arre
        self.rewards4 = rewards4_arre

        self._ptr = len(observations)
        self._size = len(observations)
        
    import numpy as np
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)

    def load_dataset_c4(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        timeouts = np.array(dataset["timeouts"], dtype=np.float32).reshape(-1, 1) 
        trajectory = np.array(dataset["trajectory"], dtype=np.float32).reshape(-1, 1)  
        step = np.array(dataset["step"], dtype=np.float32).reshape(-1, 1) 

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.timeouts = timeouts
        if sum(self.timeouts) == 0:
            self.timeouts[-1] = True  # The last one might be a truncated trajectory, manually control it
        
        next_actions = np.array(dataset["next_actions"], dtype=self.action_dtype)
        self.next_actions = next_actions
        self.trajectory = trajectory
        self.step = step

        self._ptr = len(observations)
        self._size = len(observations)


    def load_datasets(self, datasets: List[Dict[str, np.ndarray]]) -> None:
        """
        Load and merge multiple datasets simultaneously
        Args:
            datasets: List containing multiple datasets, each dataset is Dict[str, np.ndarray]
        """
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []

        for dataset in datasets:
            observations_list.append(np.array(dataset["observations"], dtype=self.obs_dtype))
            next_observations_list.append(np.array(dataset["next_observations"], dtype=self.obs_dtype))
            actions_list.append(np.array(dataset["actions"], dtype=self.action_dtype))
            rewards_list.append(np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1))
            terminals_list.append(np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1))

        # Merge all datasets
        self.observations = np.concatenate(observations_list, axis=0)
        self.next_observations = np.concatenate(next_observations_list, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        self.rewards = np.concatenate(rewards_list, axis=0)
        self.terminals = np.concatenate(terminals_list, axis=0)

        self._ptr = len(self.observations)  # Update pointer
        self._size = len(self.observations)  # Update dataset size

    def load_dataset_withNextActions(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        
        print(actions.shape)
        next_actions = np.roll(actions, -1, axis=0)
        next_actions[-1] = -actions[-1]  
        next_actions[np.where(terminals)] = -actions[np.where(terminals)]

        self.next_actions = next_actions

        self._ptr = len(observations)
        self._size = len(observations)
        
        
    def load_dataset_withNextActions_0(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        
        next_actions = np.roll(actions, -1, axis=0)
        next_actions[-1] = [0.0 for _ in actions[-1]]
        next_actions[np.where(terminals)] = -actions[np.where(terminals)]
        self.next_actions = next_actions

        self._ptr = len(observations)
        self._size = len(observations)

    def load_dataset_withTimeout(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        timeouts = np.array(dataset["timeouts"], dtype=np.float32).reshape(-1, 1) 
        trajectory = np.array(dataset["trajectory"], dtype=np.float32).reshape(-1, 1)  
        step = np.array(dataset["step"], dtype=np.float32).reshape(-1, 1) 

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.timeouts = timeouts
        if sum(self.timeouts) == 0:
            self.timeouts[-1] = True  # The last one might be a truncated trajectory, manually control it
        
        next_actions = np.array(dataset["next_actions"], dtype=self.action_dtype)
        self.next_actions = next_actions
        self.trajectory = trajectory
        self.step = step

        self._ptr = len(observations)
        self._size = len(observations)


    def load_dataset_withNextActions_0_sampleTraj(self, total_length, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        timeouts = np.array(dataset["timeouts"], dtype=np.float32).reshape(-1, 1) 

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.timeouts = timeouts
        
        next_actions = np.roll(actions, -1, axis=0)
        next_actions[-1] = [0.0 for _ in actions[-1]]
        next_actions[np.where(terminals)] = -actions[np.where(terminals)]
        self.next_actions = next_actions

        self._ptr = len(observations)
        self._size = len(observations)

        self.total_length = total_length
        self.timeouts[-1] = True  # The last one might be a truncated trajectory, manually control it
        self.sample_timeout = np.where(self.timeouts)[0]
        
        print(f"Terminals: {np.where(self.terminals)[0]}")
        print(f"timeouts: {np.where(self.timeouts)[0]}")
        print(f"Traj: {self.total_length}: {sum(self.total_length)}: {len(self.total_length)}")


    def load_dataset_arr(self, dataset: Dict[str, np.ndarray], rewards_number: int, rewards: List[np.float32]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        dataset_rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        # Create a dictionary to store all reward arrays
        rewards_arr = {}
        original_rewards_arr = {}
        
        # print(rewards_number)
        # Process each incoming reward value
        mean_rewards_arr = []
        for i, reward in rewards.items():
            # print(reward.shape)
            # print(reward)
            # print(i)
            # rewards_arr[i] = np.array(reward, dtype=np.float32).reshape(-1, 1)
            #drqi
            # original_rewards = np.array(reward, dtype=np.float32).reshape(-1, 1)
            # standard_deviation = original_rewards.std()
            # rewards_arr[i] = original_rewards - standard_deviation
            
            # normalize
            original_rewards_arr[i] = np.array(reward, dtype=np.float32).reshape(-1, 1)
            min_reward = original_rewards_arr[i].min()
            max_reward = original_rewards_arr[i].max()
            rewards_arr[i] = (original_rewards_arr[i] - min_reward) / (max_reward - min_reward)

        #     print(original_rewards[:3])
        #     print(original_rewards.std())
        #     print(rewards_arr[i][:3])
        #     # print(rewards_arr[i].shape)
            # print(i)
            # print(rewards_arr[i].shape)
            # print(rewards_arr[i].mean())
            # print(rewards_arr[i].min())
            # print(rewards_arr[i].max())
            # print(rewards_arr[i].std())
        #     # print(rewards_arr[i][:3])
        #     # print(rewards_arr[i][-20:])
        # #     mean_rewards_arr.append(rewards_arr[i].mean()) 
        # #     # print(rewards_arr[i][:100])
        # # print(np.array(mean_rewards_arr).std())
        # sys.exit()
        

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = dataset_rewards
        self.terminals = terminals

        # Store the reward array dictionary as instance variables
        # for key, arr in rewards_arr.items():
        for key, arr in rewards_arr.items():
            # setattr(self, str(key), arr)
            setattr(self, f'reward{key}', arr)
            
            # print(key)
            # print(arr.shape)
        
        # print(self.rewards)
        # print(self.reward0)
        # print("buffer.py")
        # print(self.actions.shape)
        # print(self.rewards.shape)
        # print(self.reward0.shape)
        # print(self.rewards[-5:])
        # print(self.reward0[-5:])
        # sys.exit()

        self._ptr = len(observations)
        self._size = len(observations)

        
    def load_dataset_fr(self, dataset: Dict[str, np.ndarray], fake_rewards: np.float32) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1) 
        fake_rewards_arre = np.array(fake_rewards, dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.fake_rewards = fake_rewards_arre

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def normalize_rewards(self, eps: float = 1e-3, scaling: float = 1.0) -> Tuple[float, float]:
        """
        Normalize the reward values in the experience replay buffer, with optional scaling.

        Args:
            eps (float): A small constant added to the standard deviation to prevent division by zero.
            scaling (float): Scaling factor, normalized rewards will be multiplied by this value.

        Returns:
            Tuple[float, float]: The computed mean and standard deviation of the rewards.
        """
        mean = self.rewards.mean(keepdims=True)
        std = self.rewards.std(keepdims=True) + eps
        self.rewards = (self.rewards - mean) / std * scaling
        reward_mean, reward_std = mean, std
        return reward_mean, reward_std
    
    def scaling_rewards(self, eps: float = 1e-3, scaling: float = 1.0) -> Tuple[float, float]:
        """
        Normalize the reward values in the experience replay buffer, with optional scaling.

        Args:
            eps (float): A small constant added to the standard deviation to prevent division by zero.
            scaling (float): Scaling factor, normalized rewards will be multiplied by this value.

        Returns:
            Tuple[float, float]: The computed mean and standard deviation of the rewards.
        """
        # mean = self.rewards.mean(keepdims=True)
        # std = self.rewards.std(keepdims=True) + eps
        self.rewards = self.rewards * scaling
        # reward_mean, reward_std = mean, std
        # return reward_mean, reward_std
    
    def cal_normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std
    
    def buffer_normalize_obs(self, obs_mean, obs_std, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = obs_mean
        std = obs_std + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        # obs_mean, obs_std = mean, std 
        # return obs_mean, obs_std
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        # print(self._size)
        # sys.exit()
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }

    
    # from sklearn.mixture import GaussianMixture
    
    # import copy  # Add this line to the top of the file

    def cluster_features(self, feature_matrix: torch.Tensor, method: str = 'gmm', Algo: str = '', 
                        n_clusters: int = 3, device: str = "cpu", visualize: bool = False,
                        epoch: int = 0, task: str = "hopper-expert", init_labels: np.ndarray = None,
                        initial_params: dict = None) -> List['ReplayBuffer']:
        """
        Cluster buffer data based on feature vectors using GMM or other methods.
        If init_labels is not None, it will be used to initialize GMM.
        """
        # Check if feature matrix shape matches buffer size
        if feature_matrix.shape[0] != self._size:
            raise ValueError(f"Feature matrix sample count {feature_matrix.shape[0]} does not match buffer size {self._size}")

        # Convert to numpy array and standardize
        X_numpy = feature_matrix.detach().cpu().numpy() if feature_matrix.is_cuda else feature_matrix.detach().numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numpy)

        print(X_scaled.shape)

        if init_labels is not None:
            nk = np.array([np.sum(init_labels == i) for i in range(np.max(init_labels) + 1)])
            weights_init = nk / nk.sum()
            means_init = np.array([X_scaled[init_labels == i].mean(axis=0) for i in range(np.max(init_labels) + 1)])
            precisions_init = np.array([np.eye(X_scaled.shape[1]) for _ in range(len(weights_init))])
            initial_params = {
                'weights': weights_init,
                'means': means_init,
                'precisions': precisions_init,
            }

        # Select clustering method
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X_scaled)
        elif method == 'gmm':
            # If init_labels is None, use GaussianMixture
            if initial_params is None:
                # If no initial_params provided, train a new GMM model
                gmm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=100, tol=1e-3)
                labels = gmm.fit_predict(X_scaled)  # Fully train and return labels
            else:
                # Use provided initial_params to initialize GMM model
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    weights_init=initial_params['weights'],
                    means_init=initial_params['means'],
                    precisions_init=initial_params['precisions'],
                    random_state=42,
                    max_iter=10,  # Set maximum iterations to 10
                    tol=1e-3      # Set smaller tolerance to avoid complete convergence
                )
                # Fine-tune model and predict labels, avoid complete training
                gmm.fit(X_scaled)  # Simple training
                labels = gmm.predict(X_scaled)  # Use fine-tuned model for prediction
        elif method == 'cgmm':
            cgmm = Cluster4C4(n_components=n_clusters, random_state=42, max_iter=10, tol=1e-3, device=device)
            cgmm.fit(X_scaled)
            labels = cgmm.predict(X_scaled)
        elif method == 'spectral':
            model = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X_scaled)
        elif method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_scaled)
        else:
            raise ValueError("Invalid clustering method. Choose from 'kmeans', 'gmm', 'spectral', 'agglomerative'.")




        if visualize:
            # Generate 2D embedding
            X_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                        init='pca', n_iter=1000).fit_transform(X_scaled)

            # Create save directory with date
            date_str = datetime.now().strftime("%m%d")
            save_dir = os.path.join("figure_tsne", date_str, f"{task}-c{n_clusters}/{Algo}")
            os.makedirs(save_dir, exist_ok=True)

            # Display method name changed to mathematical form C^4
            display_method = r"$C^4$"

            # Increase default font size
            plt.rcParams.update({'font.size': 14})
            title_fontsize = 22
            label_fontsize = 16
            legend_fontsize = 14
            tick_labelsize = 14

            plt.figure(figsize=(11, 7))
            colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

            for i in range(n_clusters):
                pts = X_emb[labels == i]
                plt.scatter(pts[:, 0], pts[:, 1], color=colors[i], label=f'Cluster {i}', s=40, alpha=0.85)

            # Construct title: show epoch, and show result as epoch-1
            title_text = f'{Algo}  Clusters ({display_method}, epoch={epoch-1})  tSNE'
            plt.title(title_text, fontsize=title_fontsize)

            plt.xlabel('tSNE 1', fontsize=label_fontsize)
            plt.ylabel('tSNE 2', fontsize=label_fontsize)

            plt.legend(fontsize=legend_fontsize, loc='best', frameon=True)

            # Increase tick label size
            plt.xticks(fontsize=tick_labelsize)
            plt.yticks(fontsize=tick_labelsize)

            # Save as PDF
            out_pdf = os.path.join(save_dir, f"Buffer_{method}_tSNE_{epoch}.pdf")
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight')

            plt.close()
            print(f"Saved tSNE figure to: {out_pdf}")



        # Create divided buffers
        clustered_buffers = []
        size_buffers = []
        print(labels[:100])
        
        
        for cluster_idx in range(n_clusters):
            new_buffer, n = self.build_cluster_buffer_with_traj_tail(labels=labels, cluster_idx=cluster_idx) 
            clustered_buffers.append(new_buffer)
            size_buffers.append(n)

        print(f"Clustering completed: Divided {self._size} samples into {n_clusters} clusters")
        for i, buffer in enumerate(clustered_buffers):
            print(f"Cluster {i}: {buffer._size} samples")

        return clustered_buffers, size_buffers, labels, initial_params

    

    def build_cluster_buffer_with_traj_tail(self, labels: np.ndarray, cluster_idx: int):
        N = self._size

        # Convert all to 1D
        labels_1d = np.asarray(labels[:N]).reshape(-1)
        cluster_mask = (labels_1d == cluster_idx)

        traj_1d = np.asarray(self.trajectory[:N]).reshape(-1)
        step_1d = np.asarray(self.step[:N]).reshape(-1)

        traj_ids = np.unique(traj_1d[cluster_mask])

        extra_mask = np.zeros(N, dtype=bool)
        for tid in traj_ids:
            in_traj = (traj_1d == tid)
            in_cluster_tid = cluster_mask & in_traj
            if not np.any(in_cluster_tid):
                continue
            thr = step_1d[in_cluster_tid].max()
            extra_mask |= in_traj & (step_1d > thr)
            # extra_mask |= in_traj & (step_1d != thr)

        combined_mask = cluster_mask | extra_mask
        idx = np.flatnonzero(combined_mask)
        n = idx.size

        # New buffer uses original capacity, first n slots filled
        new_buffer = ReplayBuffer(
            buffer_size=self._max_size,
            obs_shape=self.obs_shape,
            obs_dtype=self.obs_dtype,
            action_dim=self.action_dim,
            action_dtype=self.action_dtype,
            device=self.device.type
        )

        new_buffer.observations       = self.observations[idx]
        new_buffer.next_observations  = self.next_observations[idx]
        new_buffer.actions            = self.actions[idx]
        new_buffer.rewards            = self.rewards[idx]
        new_buffer.terminals          = self.terminals[idx]
        new_buffer.trajectory         = self.trajectory[idx]
        new_buffer.step               = self.step[idx]

        new_buffer._size = n
        new_buffer._ptr  = n % new_buffer._max_size

        return new_buffer, n




    def sample_st(self, batch_size: int) -> Dict[str, torch.Tensor]:

        
        random_value = np.random.randint(0, self._size)
        # Find the position of random_value in self.sample_timeout
        index = np.searchsorted(self.sample_timeout, random_value, side='right') - 1
        # Get upper and lower bounds
        lower_bound = self.sample_timeout[index] if index + 1 > 0 else 0
        upper_bound = self.sample_timeout[index + 1] if index + 1 < len(self.sample_timeout) else self.sample_timeout[-1]
        batch_indexes = np.random.randint(lower_bound,upper_bound+1, size=min(batch_size,upper_bound+1-lower_bound))
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }
        
    def sample_withNextActions(self, batch_size: int) -> Dict[str, torch.Tensor]:

        # batch_size = len(self.observations)
        # print(self._size)
        # print(self._ptr)
        # print(self._max_size)
        # print(batch_size)
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        # print(batch_indexes[:10])

        # print(self._size)
        # sys.exit()
        # Check if there is a timeouts attribute, if not use terminals values
        timeouts = (
            torch.tensor(self.timeouts[batch_indexes]).to(self.device)
            if hasattr(self, "timeouts") and self.timeouts is not None
            else torch.where(torch.tensor(self.terminals[batch_indexes]).to(self.device) == 1, 1, 0)
        )

        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "next_actions": torch.tensor(self.next_actions[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "timeouts": timeouts,
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "trajectory": torch.tensor(self.trajectory[batch_indexes]).to(self.device),
            "step": torch.tensor(self.step[batch_indexes]).to(self.device),
        }
        
    def sample_arr(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        sample_dict = {  
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),  
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),  
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),  
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device), 
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }  
        # print('here')
        # print(self.reward9)
        # Assuming there are self.num_rewards rewards
        # print(self.num_rewards) 
        for i in range(self.num_rewards):  
            reward_key = f"reward{i}"  
            # print(getattr(self, reward_key))
            # print('here')
            sample_dict[reward_key] = torch.tensor(getattr(self, reward_key)[batch_indexes]).to(self.device)  
        # sys.exit()
        return sample_dict
        
    
    # def sample_all(self) -> Dict[str, np.ndarray]:
    #     return {
    #         "observations": self.observations[:self._size].copy(),
    #         "actions": self.actions[:self._size].copy(),
    #         "next_observations": self.next_observations[:self._size].copy(),
    #         "terminals": self.terminals[:self._size].copy(),
    #         "rewards": self.rewards[:self._size].copy(),
    #     }

    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": torch.tensor(self.observations[:self._size]).to(self.device),
            "actions": torch.tensor(self.actions[:self._size]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[:self._size]).to(self.device),
            "next_actions": torch.tensor(self.next_actions[:self._size]).to(self.device),
            "terminals": torch.tensor(self.terminals[:self._size]).to(self.device),
            # "timeouts": timeouts,
            # "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            # "trajectory": torch.tensor(self.trajectory[batch_indexes]).to(self.device),
            # "step": torch.tensor(self.step[batch_indexes]).to(self.device),
        }
    
    def sample_c4(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        timeouts = (
            torch.tensor(self.timeouts[batch_indexes]).to(self.device)
            if hasattr(self, "timeouts") and self.timeouts is not None
            else torch.where(torch.tensor(self.terminals[batch_indexes]).to(self.device) == 1, 1, 0)
        )

        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "next_actions": torch.tensor(self.next_actions[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "timeouts": timeouts,
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "trajectory": torch.tensor(self.trajectory[batch_indexes]).to(self.device),
            "step": torch.tensor(self.step[batch_indexes]).to(self.device),
        }

    def samplen(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards1": torch.tensor(self.rewards1[batch_indexes]).to(self.device),
            "rewards2": torch.tensor(self.rewards2[batch_indexes]).to(self.device),
        }
    
    def sample5(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards0": torch.tensor(self.rewards0[batch_indexes]).to(self.device),
            "rewards1": torch.tensor(self.rewards1[batch_indexes]).to(self.device),
            "rewards2": torch.tensor(self.rewards2[batch_indexes]).to(self.device),
            "rewards3": torch.tensor(self.rewards3[batch_indexes]).to(self.device),
            "rewards4": torch.tensor(self.rewards4[batch_indexes]).to(self.device),
        }
        
    def sample5_gtr(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards0": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards1": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards2": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards3": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "rewards4": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }
        
    def sample_fr(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        # print("sample_fr")
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
            "fake_rewards": torch.tensor(self.fake_rewards[batch_indexes]).to(self.device),
        }
    
    def samplen_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),
            "rewards1": self.rewards1[:self._size].copy(),
            "rewards2": self.rewards2[:self._size].copy(),
        }
        
    def sample_buffer(self, batch_size: int) -> Dict[str, torch.Tensor]:
        # Sample a batch from the buffer
        batch = self.sample(batch_size)

        # Create a new buffer with the same parameters as the original buffer
        new_buffer = ReplayBuffer(
            # num_rewards = self.num_rewards,
            buffer_size=self._max_size,
            obs_shape=self.obs_shape,
            obs_dtype=self.obs_dtype,
            action_dim=self.action_dim,
            action_dtype=self.action_dtype,
            device=self.device.type
        )

        # Add the sampled batch to the new buffer
        new_buffer.add_batch(
            obss=batch["observations"].cpu().numpy(),
            next_obss=batch["next_observations"].cpu().numpy(),
            actions=batch["actions"].cpu().numpy(),
            rewards=batch["rewards"].cpu().numpy(),
            terminals=batch["terminals"].cpu().numpy()
        )

        return new_buffer

    def sample_Traj2Buffer(self, traj_length_list, episode_end_list,  batch_size: int) -> Dict[str, torch.Tensor]:
        # Sample a batch from the buffer
        batch = self.sample_Traj(traj_length_list, episode_end_list,  batch_size)

        # Create a new buffer with the same parameters as the original buffer
        new_buffer = ReplayBuffer(
            # num_rewards = self.num_rewards,
            buffer_size=self._max_size,
            obs_shape=self.obs_shape,
            obs_dtype=self.obs_dtype,
            action_dim=self.action_dim,
            action_dtype=self.action_dtype,
            device=self.device.type
        )

        # Add the sampled batch to the new buffer
        new_buffer.add_batch(
            obss=batch["observations"].cpu().numpy(),
            next_obss=batch["next_observations"].cpu().numpy(),
            actions=batch["actions"].cpu().numpy(),
            rewards=batch["rewards"].cpu().numpy(),
            terminals=batch["terminals"].cpu().numpy()
        )

        return new_buffer
        return new_buffer

    def sample_Traj(self, traj_length_list, episode_end_list,  batch_size: int) -> Dict[str, torch.Tensor]:

        num_traj = len(traj_length_list)
        sum_traj = 0
        traj_int_list = []
        while sum_traj < batch_size:  
            i = random.randint(0, num_traj-1)  
            traj_start = max (0, episode_end_list[i]-traj_length_list[i])
            traj_end = min(episode_end_list[-1], episode_end_list[i])
            traj_int = [i for i in range(traj_start, traj_end)]
            traj_int_list.extend(traj_int)

        sum_traj += traj_length_list[i]

        if len(traj_int_list) > batch_size:
            batch_indexes = traj_int_list[:batch_size]
        print("len(batch_indexes)"+str(len(batch_indexes)))

        

        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }
