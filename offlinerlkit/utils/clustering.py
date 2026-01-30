import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns
import sys
import os
import torch

def find_L1(local_dataset, obs_mean, obs_std, device="cpu", threshold=0.1):
    import torch

    X = local_dataset["next_observations"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    # Compute L1 distance matrix
    def compute_l1_distance_matrix(x_tensor):
        diff = x_tensor[:, None, :] - x_tensor[None, :, :]  # [n, n, d]
        l1_dist = diff.abs().sum(dim=2)  # [n, n]
        return l1_dist

    l1_distance_matrix = compute_l1_distance_matrix(x_tensor)

    # Mask the diagonal (set to infinity to avoid selecting itself)
    n = l1_distance_matrix.shape[0]
    mask = torch.eye(n, dtype=torch.bool, device=device)  # Mask with True on diagonal
    l1_distance_matrix.masked_fill_(mask, float('inf'))  # Set diagonal to inf

    # Generate row and column index grid
    rows, cols = torch.meshgrid(torch.arange(n, device=device), 
                            torch.arange(n, device=device), 
                            indexing='ij')

    # Extract upper triangular part (excluding diagonal)
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    l1_dist_triu = l1_distance_matrix[triu_mask]  # Upper triangular part flattened to 1D vector
    i_triu = rows[triu_mask]  # Corresponding row indices
    j_triu = cols[triu_mask]  # Corresponding column indices

    # Calculate the proportion of L1 distances < threshold
    num_pairs = l1_dist_triu.shape[0]  # Total pairs = n*(n-1)/2
    num_close_pairs = (l1_dist_triu < threshold).sum().item()
    close_ratio = num_close_pairs / num_pairs

    # Calculate the proportion of L1 distances < threshold and |i-j| > 1000
    index_distance_threshold = 1000
    far_close_mask = (l1_dist_triu < threshold) & ((i_triu - j_triu).abs() > index_distance_threshold)
    num_far_close_pairs = far_close_mask.sum().item()
    far_close_ratio = num_far_close_pairs / num_pairs

    # Find the top 5 smallest L1 distances and their indices
    k = 10
    min_values, min_indices = torch.topk(-l1_dist_triu, k)  # Use negative to simulate taking minimum
    min_values = -min_values  # Restore correct distance values
    min_i = i_triu[min_indices]
    min_j = j_triu[min_indices]

    # Find the top 5 largest L1 distances and their indices
    max_values, max_indices = torch.topk(l1_dist_triu, k)
    max_i = i_triu[max_indices]
    max_j = j_triu[max_indices]

    # Get all (i,j) pairs that satisfy L1 < threshold
    close_mask = l1_dist_triu < threshold
    close_pairs = list(zip(i_triu[close_mask].cpu().numpy(), 
                          j_triu[close_mask].cpu().numpy()))

    # Return all (i,j) pairs that satisfy L1 < threshold
    return close_pairs
    

def cluster_and_visualize(local_dataset, method='kmeans', k=3, device="cpu", n=5):


    if method == 'ogmm':
        return cluster_and_visualize_ogmm(local_dataset, method, k, device, n)
    elif method == 'none':
        return cluster_and_visualize_none(local_dataset, method, k, device, n)
    else:
        return cluster_and_visualize_model(local_dataset, method, k, device, n)
    
def cluster_and_visualize_none(local_dataset, method='kmeans', k=3, device="cpu", n=5):
    divided_datasets = []
    for _ in range(k):
        divided_datasets.append(local_dataset)
    return k, {}, [], divided_datasets


# def cluster_and_visualize_ogmm(local_dataset, method='kmeans', k=3, device="cpu", n=5):
def cluster_and_visualize_ogmm(local_dataset, method='ogmm', k=3, device="cpu", n=5, variance_reduction_threshold=0.0):
    """
    True overlapping GMM implementation with GPU acceleration
    
    Parameters:
        device: "cuda" or "cpu"
        variance_reduction_threshold: Minimum variance reduction ratio to allow overlap
    """
    # Data preparation
    s = local_dataset["observations"]
    a = local_dataset["actions"]
    X = np.hstack([s, a])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensor and move to GPU
    if device == "cuda" and torch.cuda.is_available():
        X_tensor = torch.tensor(X_scaled, device="cuda")
    else:
        X_tensor = torch.tensor(X_scaled, device="cpu")
    
    # Train GMM model (sklearn doesn't support GPU yet, keep CPU)
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    
    # Get initial clustering results
    probs = gmm.predict_proba(X_scaled)
    main_labels = np.argmax(probs, axis=1)
    
    # Initialize cluster assignment matrix (using GPU)
    cluster_assignments = torch.zeros((len(X), k), dtype=torch.bool, device=device)
    cluster_assignments[torch.arange(len(X)), torch.tensor(main_labels, device=device)] = True
    
    # Calculate initial cluster variances (GPU accelerated)
    cluster_variances = []
    for i in range(k):
        mask = cluster_assignments[:, i]
        if mask.sum() > 1:
            cluster_var = torch.var(X_tensor[mask], dim=0).mean().item()
            cluster_variances.append(cluster_var)
        else:
            cluster_variances.append(0)
    
    max_epochs = n
    prev_assignments = cluster_assignments.clone()
    no_improvement_count = 0
    has_converged = False
    early_stop_patience = 2
    
    for epoch in range(max_epochs):
        if has_converged:
            break
            
        epoch_changes = 0
        for point_idx in range(len(X)):
            point = X_tensor[point_idx]
            original_cluster = main_labels[point_idx]
            
            for target_cluster in range(k):
                if target_cluster == original_cluster:
                    continue
                
                # Check current assignment status
                current_assignment = cluster_assignments[point_idx, target_cluster]
                if current_assignment:  # Skip if already assigned
                    continue
                
                # Current target cluster points and variance
                target_mask = cluster_assignments[:, target_cluster]
                if target_mask.sum() == 0:
                    continue
                
                # Calculate current variance
                current_var = torch.var(X_tensor[target_mask], dim=0).mean().item()
                
                # Simulate adding point to target cluster
                temp_mask = target_mask.clone()
                temp_mask[point_idx] = True
                new_var = torch.var(X_tensor[temp_mask], dim=0).mean().item()
                
                # Calculate variance reduction ratio
                if current_var > 0:
                    var_reduction = (current_var - new_var) / current_var
                else:
                    var_reduction = 0
                
                # Determine whether to assign
                if var_reduction > variance_reduction_threshold:
                    cluster_assignments[point_idx, target_cluster] = True
                    cluster_variances[target_cluster] = new_var
                    epoch_changes += 1
        
        # Convergence detection
        if epoch_changes == 0:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        # Check for early stopping
        if no_improvement_count >= early_stop_patience:
            has_converged = True
        
        # Check for complete convergence
        if torch.equal(prev_assignments, cluster_assignments):
            break
            
        prev_assignments = cluster_assignments.clone()
    
    # Subsequent processing and visualization
    
    # Convert back to CPU numpy array
    cluster_assignments = cluster_assignments.cpu().numpy()
    
    # Build overlapping datasets
    divided_datasets = []
    total_points = 0
    for cluster_i in range(k):
        cluster_indices = np.where(cluster_assignments[:, cluster_i])[0]
        cluster_data = {key: value[cluster_indices] for key, value in local_dataset.items()}
        divided_datasets.append(cluster_data)
        total_points += len(cluster_indices)
        
    
    # Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    # Plot non-overlapping points
    for i in range(k):
        pure_mask = cluster_assignments[:, i] & (np.sum(cluster_assignments, axis=1) == 1)
        plt.scatter(X_pca[pure_mask, 0], X_pca[pure_mask, 1], 
                   color=colors[i], label=f'Cluster {i}', alpha=0.6)
    
    # Plot overlapping points (marked in black)
    overlap_mask = np.sum(cluster_assignments, axis=1) > 1
    plt.scatter(X_pca[overlap_mask, 0], X_pca[overlap_mask, 1], 
               color='black', marker='x', s=50, label='Overlapping Points')
    
    plt.title(f'Overlapping GMM (k={k}, threshold={variance_reduction_threshold})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    plt.savefig(f"{method}.png")

    return k, main_labels, [], divided_datasets




def cluster_and_visualize_model(local_dataset, method='kmeans', k=3, device="cpu", n=5):
    """
    Perform clustering on local_dataset and visualize results.

    Parameters:
        local_dataset (dict): Dictionary containing "observations" and "actions".
        method (str): Clustering method, options: 'kmeans', 'gmm', 'spectral', 'agglomerative'. 
        k (int): Number of clusters to specify.

    Returns:
        best_k (int): Optimal number of clusters.
        best_labels (np.array): Cluster labels for each sample.
        hulls (list): Convex hull information for each cluster (empty list here).
        divided_datasets (list): List of divided datasets.
    """
    # Extract s and a
    s = local_dataset["observations"]  # s is 7-dimensional
    a = local_dataset["actions"]  # a is 3-dimensional
    next_s = local_dataset["next_observations"]  # next_s is 7-dimensional
    # X = s
    X = np.hstack([s, a])

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'kmeans':
        # Use KMeans for clustering
        model = KMeans(n_clusters=k, random_state=42)
    elif method == 'gmm' or method == 'cgmm':
        # Use Gaussian Mixture Model for clustering
        model = GaussianMixture(n_components=k, random_state=42)
    elif method == 'bgmm':
        # Use Gaussian Mixture Model for clustering
        model = BayesianGaussianMixture(n_components=k, random_state=42)
    elif method == 'spectral':
        # Use Spectral Clustering
        model = SpectralClustering(n_clusters=k, random_state=42)
    elif method == 'agglomerative':
        # Use Hierarchical Clustering
        model = AgglomerativeClustering(n_clusters=k)
    elif method == 'ward' : 
        # Use Ward method for hierarchical clustering 
        model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'gmm', 'spectral', 'agglomerative'.")

    labels = model.fit_predict(X_scaled)
    best_k = k
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # Visualize dimensionality-reduced data
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, best_k))
    for i in range(best_k):
        # Get current cluster's PCA dimensionality reduction results
        cluster_pca = X_pca[labels == i]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[i], label=f'Cluster {i + 1}')
    plt.title(f'Clusters ({method}, k={best_k}) - PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(f"{method}.png")

    divided_datasets = []
    for cluster_idx in range(best_k):
        cluster_mask = labels == cluster_idx
        cluster_dataset = {}
        for key, value in local_dataset.items():
            cluster_dataset[key] = value[cluster_mask]
        divided_datasets.append(cluster_dataset)

    if len(local_dataset["rewards"]) < 2e5:
            # Call function for subsequent trajectory data processing
        divided_datasets = process_datasets_subsequent(local_dataset, divided_datasets, device, n)
        # Call function for previous trajectory data processing
        divided_datasets = process_datasets_previous(local_dataset, divided_datasets, device, n)
    else:
        if n > 0:       
            device = "cpu"
            divided_datasets = combine_datasets_trajectory(local_dataset, divided_datasets ,device, n)

    best_labels = labels
    hulls = []  # No convex hull needed, so empty list here



    return best_k, best_labels, hulls, divided_datasets


def combine_datasets_trajectory(local_dataset, divided_datasets, device, extend_traj, batch_size=1000):
    # Convert local_dataset data to PyTorch tensors and move to GPU
    local_tensors = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in local_dataset.items()}

    for cluster_dataset in tqdm(divided_datasets, desc="Merging trajectories", unit="dataset"):
        # Convert cluster_dataset trajectory and step to PyTorch tensors
        cluster_traj = torch.tensor(cluster_dataset["trajectory"], dtype=torch.int64, device=device)
        
        # Find all unique trajectory indices in cluster_dataset (deduplicated)
        unique_trajectories = torch.unique(cluster_traj)
        
        # Find all data points in local_dataset that belong to these trajectories
        mask = torch.isin(local_tensors["trajectory"], unique_trajectories)
        
        # Add matching trajectory data to cluster_dataset
        for key in cluster_dataset.keys():
            new_data = local_tensors[key][mask].cpu().numpy()
            if cluster_dataset[key].ndim == 1:
                cluster_dataset[key] = np.hstack((cluster_dataset[key], new_data.flatten()))
            else:
                cluster_dataset[key] = np.vstack((cluster_dataset[key], new_data))

        # Deduplicate + sort
        if len(cluster_dataset["trajectory"]) > 0:
            # 1. Construct structured array for deduplication
            traj_step = np.core.records.fromarrays(
                [cluster_dataset["trajectory"], cluster_dataset["step"]],
                names="trajectory,step"
            )
            _, unique_indices = np.unique(traj_step, return_index=True)
            
            # 2. Sort by (trajectory, step) after deduplication
            # Get deduplicated trajectory and step
            traj_after_dedup = cluster_dataset["trajectory"][unique_indices]
            step_after_dedup = cluster_dataset["step"][unique_indices]
            
            # Generate sort indices: group by trajectory first, then ascending step
            sort_indices = np.lexsort((step_after_dedup, traj_after_dedup))
            
            # Apply deduplication and sorting to all fields
            for key in cluster_dataset.keys():
                if cluster_dataset[key].ndim == 1:
                    cluster_dataset[key] = cluster_dataset[key][unique_indices][sort_indices]
                else:
                    cluster_dataset[key] = cluster_dataset[key][unique_indices, :][sort_indices, :]


    return divided_datasets


def process_datasets_subsequent(local_dataset, divided_datasets, device, extend_traj, batch_size=1000):
    # Convert local_dataset data to PyTorch tensors and move to GPU
    local_tensors = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in local_dataset.items()}

    for i in tqdm(range(extend_traj), desc="Processing subsequent data", unit="iter"):
        for cluster_dataset in divided_datasets:
            # Convert cluster_dataset observations to PyTorch tensors and move to GPU
            cluster_obs = torch.tensor(cluster_dataset["observations"], dtype=torch.float32, device=device)

            # Check if adjacent observations in local_dataset are equal
            obs_local_dataset = torch.all(local_tensors["observations"][1:] == local_tensors["next_observations"][:-1], dim=1)

            # Check in batches whether local_dataset[i] is in cluster_dataset
            obs_i_in_cluster = torch.zeros(len(local_tensors["observations"][:-1]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][:-1].unsqueeze(1), dim=2), dim=1)
                obs_i_in_cluster |= batch_result

            # Check in batches whether local_dataset[i+1] is not in cluster_dataset
            obs_i_next_in_cluster = torch.zeros(len(local_tensors["observations"][1:]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][1:].unsqueeze(1), dim=2), dim=1)
                obs_i_next_in_cluster |= batch_result

            # Filter indices that satisfy the conditions
            valid_indices = (obs_local_dataset & obs_i_in_cluster & ~obs_i_next_in_cluster).nonzero(as_tuple=True)[0]

            # Add matching local_dataset data to cluster_dataset
            for key in cluster_dataset.keys():
                new_data = local_tensors[key][valid_indices + 1].cpu().numpy()
                if cluster_dataset[key].ndim == 1:
                    cluster_dataset[key] = np.hstack((cluster_dataset[key], new_data.flatten()))
                else:
                    cluster_dataset[key] = np.vstack((cluster_dataset[key], new_data))

    return divided_datasets

def process_datasets_previous(local_dataset, divided_datasets, device, extend_traj, batch_size=1000):
    # Convert local_dataset data to PyTorch tensors and move to GPU
    local_tensors = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in local_dataset.items()}

    for i in tqdm(range(extend_traj), desc="Processing previous data", unit="iter"):
        for cluster_dataset in divided_datasets:
            # Convert cluster_dataset observations to PyTorch tensors and move to GPU
            cluster_obs = torch.tensor(cluster_dataset["observations"], dtype=torch.float32, device=device)

            # Check if adjacent observations in local_dataset are equal
            obs_local_dataset = torch.all(local_tensors["observations"][1:] == local_tensors["next_observations"][:-1], dim=1)

            # Check in batches whether local_dataset[i] is in cluster_dataset
            obs_i_in_cluster = torch.zeros(len(local_tensors["observations"][1:]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][1:].unsqueeze(1), dim=2), dim=1)
                obs_i_in_cluster |= batch_result

            # Check in batches whether local_dataset[i-1] is not in cluster_dataset
            obs_i_prev_in_cluster = torch.zeros(len(local_tensors["observations"][:-1]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][:-1].unsqueeze(1), dim=2), dim=1)
                obs_i_prev_in_cluster |= batch_result

            # Filter indices that satisfy the conditions
            valid_indices = (obs_local_dataset & obs_i_in_cluster & ~obs_i_prev_in_cluster).nonzero(as_tuple=True)[0]

            # Add matching local_dataset data to cluster_dataset
            for key in cluster_dataset.keys():
                new_data = local_tensors[key][valid_indices].cpu().numpy()
                if cluster_dataset[key].ndim == 1:
                    cluster_dataset[key] = np.hstack((cluster_dataset[key], new_data.flatten()))
                else:
                    cluster_dataset[key] = np.vstack((cluster_dataset[key], new_data))

    return divided_datasets






def clustering(X, method='kmeans', k=3, device="cpu", n=5):

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'kmeans':
        # Use KMeans for clustering
        model = KMeans(n_clusters=k, random_state=42)
    elif method == 'gmm':
        # Use Gaussian Mixture Model for clustering
        model = GaussianMixture(n_components=k, random_state=42)
    elif method == 'bgmm':
        # Use Gaussian Mixture Model for clustering
        model = BayesianGaussianMixture(n_components=k, random_state=42)
    elif method == 'spectral':
        # Use Spectral Clustering
        model = SpectralClustering(n_clusters=k, random_state=42)
    elif method == 'agglomerative':
        # Use Hierarchical Clustering
        model = AgglomerativeClustering(n_clusters=k)
    elif method == 'ward' : 
        # Use Ward method for hierarchical clustering 
        model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'gmm', 'spectral', 'agglomerative'.")

    labels = model.fit_predict(X_scaled)
    best_k = k
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # Visualize dimensionality-reduced data
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, best_k))
    for i in range(best_k):
        # Get current cluster's PCA dimensionality reduction results
        cluster_pca = X_pca[labels == i]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[i], label=f'Cluster {i + 1}')
    plt.title(f'Clusters ({method}, k={best_k}) - PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(f"{method}.png")

    divided_datasets = []
    for cluster_idx in range(best_k):
        cluster_mask = labels == cluster_idx
        cluster_dataset = {}
        for key, value in local_dataset.items():
            cluster_dataset[key] = value[cluster_mask]
        divided_datasets.append(cluster_dataset)



    return divided_datasets
