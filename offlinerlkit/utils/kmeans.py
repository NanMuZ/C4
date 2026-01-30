import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def cluster_and_visualize(local_dataset, max_k=10):
    """
    Cluster local_dataset and visualize results.
    
    Parameters:
        local_dataset (dict): Dictionary containing "observations" and "actions".
        max_k (int): Maximum number of clusters, used to dynamically determine k range.
    
    Returns:
        best_k (int): Best number of clusters.
        best_labels (np.array): Cluster labels for each sample.
        hulls (list): Convex hull information for each cluster.
    """
    # Extract s and a
    s = local_dataset["observations"]  # s is 7-dimensional
    a = local_dataset["actions"]       # a is 3-dimensional
    # X = np.hstack([s, a])  # Merge into 10-dimensional data
    X = s

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dynamically determine k range
    def determine_k_range(X, max_k):
        silhouette_scores = []
        inertia_values = []
        range_k = range(2, max_k + 1)
        
        for k in range_k:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            inertia_values.append(kmeans.inertia_)
        
        # Select k range based on silhouette score
        best_k_silhouette = range_k[np.argmax(silhouette_scores)]
        k_range_silhouette = range(max(2, best_k_silhouette - 2), min(max_k, best_k_silhouette + 3))
        
        # Select k range based on elbow method
        best_k_elbow = range_k[np.argmin(np.diff(inertia_values, 2)) + 1]  # Second-order difference minimum
        k_range_elbow = range(max(2, best_k_elbow - 2), min(max_k, best_k_elbow + 3))
        
        # Combine both methods
        k_range = sorted(set(k_range_silhouette).union(set(k_range_elbow)))
        return list(k_range)

    range_n_clusters = determine_k_range(X_scaled, max_k)

    # Find best k value (based on convex hull distance)
    def compute_average_min_distance(X, labels, k):
        min_distances = []
        for i in range(k):
            for j in range(i + 1, k):
                points_i = X[labels == i]
                points_j = X[labels == j]
                if len(points_i) == 0 or len(points_j) == 0:
                    continue
                dist_matrix = distance.cdist(points_i, points_j, 'euclidean')
                min_dist = np.min(dist_matrix)
                min_distances.append(min_dist)
        return np.mean(min_distances) if min_distances else 0

    best_avg_distance = -1
    best_k = 2
    best_labels = None

    # Add progress bar for finding best k loop
    for k in tqdm(range_n_clusters, desc="Finding best k"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        avg_distance = compute_average_min_distance(X_scaled, labels, k)
        if avg_distance > best_avg_distance:
            best_avg_distance = avg_distance
            best_k = k
            best_labels = labels

    # Cluster using best k value
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate convex hulls
    hulls = []
    # Add progress bar for computing convex hulls loop
    for i in tqdm(range(best_k), desc="Computing convex hulls"):
        cluster_points = X_scaled[labels == i]
        if len(cluster_points) >= cluster_points.shape[1] + 1:
            hulls.append((cluster_points, None))
        else:
            hulls.append((cluster_points, None))

    # Dimensionality reduction visualization (using PCA to reduce to 2 dimensions)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Visualize dimensionally reduced data
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, best_k))
    for i, (cluster_points, hull) in enumerate(hulls):
        # Get PCA dimensionality reduction result for current cluster
        cluster_pca = X_pca[labels == i]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[i], label=f'Cluster {i + 1}')
        if hull is not None:
            # Calculate convex hull projection in PCA space
            hull_points = cluster_points[hull.vertices]
            hull_pca = pca.transform(hull_points)
            hull_pca = np.vstack((hull_pca, hull_pca[0]))  # Close convex hull
            plt.plot(hull_pca[:, 0], hull_pca[:, 1], color=colors[i], linewidth=2, linestyle='--')
    plt.title(f'Clusters with Convex Hulls (k={best_k}) - PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    plt.savefig("kmeans.png")

    divided_datasets = []
    for cluster_idx in range(best_k):
        cluster_mask = best_labels == cluster_idx
        cluster_dataset = {}
        for key, value in local_dataset.items():
            cluster_dataset[key] = value[cluster_mask]
        divided_datasets.append(cluster_dataset)

    return best_k, best_labels, hulls, divided_datasets
