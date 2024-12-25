import geopandas
import numpy as np
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
import csv

# Model: GPT-4, OpenAI
# Prompts Used:
# - Asked for help identifying issues in hierarchical agglomerative clustering implementation.
# - Requested guidance on adjusting distance matrix operations to handle dynamic cluster merging.
# - Clarified expected output formats and types for normalization functions.
# - Received suggestions on resolving indexing errors and type mismatches during distance calculations.
# - Verified clustering visualization steps and proper return types for function outputs.

# 0.1 load_data(filepath)
def load_data(filepath):
    data = []
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

# 0.2 calc_features(row)
def calc_features(row):
    # Extracting 6 statistics & converting float64
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)

# 0.3 normalize_features(features)
def normalize_features(features):
    features = np.array(features)  # Convert NumPy array easier manipulation
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    
    # Apply min-max normalization
    normalized = (features - min_vals) / (max_vals - min_vals)
    
    # Return normalized features as list of lists
    return normalized.tolist()

# 0.4 hac(features)
def hac(features):
    # Ensure features is a NumPy array (in case it is not already)
    features = np.array(features)
    n = len(features)
    
    # Initialize the distance matrix as a dictionary
    dist_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Euclidean distance between feature vectors
            dist_matrix[(i, j)] = np.linalg.norm(features[i] - features[j])

    # Dictionary to keep track of clusters and active clusters set
    clusters = {i: [i] for i in range(n)}
    active_clusters = set(range(n))  # Keep track of active clusters
    Z = []  # To store the linkage matrix

    # Step counter to create new cluster indices
    new_cluster_idx = n

    for step in range(n - 1):
        # Find the closest pair of clusters that have not been merged
        min_dist = np.inf
        cluster_pair = (-1, -1)

        for (i, j), distance in dist_matrix.items():
            if distance < min_dist:
                min_dist = distance
                cluster_pair = (i, j)

        # Get the indices of the two clusters to merge
        i, j = cluster_pair
        if i > j:  # Ensure i < j for consistency
            i, j = j, i

        # Merge clusters i and j into a new cluster
        distance = dist_matrix[(i, j)]
        cluster_size = len(clusters[i]) + len(clusters[j])

        # Append the merge information to Z
        Z.append([i, j, distance, cluster_size])

        # Create a new cluster index and update clusters
        clusters[new_cluster_idx] = clusters[i] + clusters[j]

        # Remove the merged clusters from active_clusters set
        active_clusters.remove(i)
        active_clusters.remove(j)

        # Add the new cluster index to active_clusters
        active_clusters.add(new_cluster_idx)

        # Update the distance matrix for the new cluster
        for k in active_clusters:
            if k != new_cluster_idx:
                # Calculate the distance of the new cluster to other clusters using single linkage
                dist_matrix[(min(new_cluster_idx, k), max(new_cluster_idx, k))] = min(
                    dist_matrix.get((min(i, k), max(i, k)), np.inf),
                    dist_matrix.get((min(j, k), max(j, k)), np.inf)
                )

        # Remove old distances involving the merged clusters
        dist_matrix = {
            (key_i, key_j): value
            for (key_i, key_j), value in dist_matrix.items()
            if key_i not in {i, j} and key_j not in {i, j}
        }

        # Increment the new cluster index for the next merge
        new_cluster_idx += 1

    return np.array(Z)

# 0.5 fig_hac(Z, names)
def fig_hac(Z, names):
    fig = plt.figure()  # Create a figure object
    dendrogram(Z, labels=names, leaf_rotation=90)  # Generate dendrogram
    plt.tight_layout()  # Adjust layout
    plt.show()  # Show plot
    return fig  # Return figure object

if __name__ == "__main__":
    # Example test setup
    test_data = [
        {'Population': '100', 'Net migration': '0.2', 'GDP ($ per capita)': '20000', 'Literacy (%)': '95', 'Phones (per 1000)': '500', 'Infant mortality (per 1000 births)': '5'},
        {'Population': '150', 'Net migration': '0.3', 'GDP ($ per capita)': '21000', 'Literacy (%)': '90', 'Phones (per 1000)': '600', 'Infant mortality (per 1000 births)': '6'},
        {'Population': '200', 'Net migration': '0.1', 'GDP ($ per capita)': '22000', 'Literacy (%)': '85', 'Phones (per 1000)': '400', 'Infant mortality (per 1000 births)': '7'}
    ]

    # Calculate features and normalize them
    features = [calc_features(row) for row in test_data]
    normalized_features = normalize_features(features)
    
    # Perform HAC
    Z = hac(normalized_features)
    print("Linkage Matrix Z:")
    print(Z)
