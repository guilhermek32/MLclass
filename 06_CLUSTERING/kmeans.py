# %% [markdown]
# # Improved K-Means Clustering Algorithm
# This script refactors the original code into a more structured, robust, and insightful workflow.

# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

# %%
def find_optimal_k(scaled_data: pd.DataFrame, max_k: int = 10):
    """
    Finds the optimal number of clusters (k) using the Elbow and Silhouette methods.
    
    Args:
        scaled_data (pd.DataFrame): The preprocessed and scaled data.
        max_k (int): The maximum number of clusters to test.

    Returns:
        int: The suggested optimal number of clusters based on the highest silhouette score.
    """
    inertia_values = []
    silhouette_scores_list = []
    K = range(2, max_k + 1) # Start from 2 for silhouette score

    print("Finding optimal k...")
    for k in K:
        kmeans = KMedoids(n_clusters=k, random_state=42, init='heuristic')
        kmeans.fit(scaled_data)
        inertia_values.append(kmeans.inertia_)
        score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores_list.append(score)

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Elbow Method Plot
    ax1.plot(K, inertia_values, 'bo-')
    ax1.set_title('Elbow Method For Optimal k')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_xticks(list(K))

    # Silhouette Score Plot
    ax2.plot(K, silhouette_scores_list, 'ro-')
    ax2.set_title('Silhouette Score For Various k')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_xticks(list(K))
    
    plt.suptitle('Cluster Evaluation Metrics', fontsize=16)
    plt.show()

    # Suggest the best k based on the highest silhouette score
    optimal_k = K[np.argmax(silhouette_scores_list)]
    print(f"\nSuggested optimal k based on highest silhouette score: {optimal_k}")
    return optimal_k

# %%
def visualize_clusters_pca(scaled_data: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray, original_features: list):
    """
    Visualizes clusters using PCA for dimensionality reduction.
    This is more robust than plotting only the first two features.
    
    Args:
        scaled_data (pd.DataFrame): The scaled data used for clustering.
        labels (np.ndarray): The cluster labels for each data point.
        centroids (np.ndarray): The cluster centroids.
        original_features (list): Names of the original features for context.
    """
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(scaled_data)
    centroids_pca = pca.transform(centroids)

    # Create a DataFrame for easier plotting with seaborn
    df_pca = pd.DataFrame(data_pca, columns=['Principal Component 1', 'Principal Component 2'])
    df_pca['Cluster'] = labels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='Principal Component 1', 
        y='Principal Component 2', 
        hue='Cluster', 
        data=df_pca, 
        palette='viridis', 
        s=100, 
        alpha=0.7,
        legend='full'
    )
    plt.scatter(
        centroids_pca[:, 0], 
        centroids_pca[:, 1], 
        c='black', 
        s=250, 
        marker='X', 
        label='Centroids'
    )
    
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    plt.title(f'K-Means Clustering Visualization (via PCA)\nExplained Variance: {explained_variance:.2f}%')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
def main():
    """
    Main function to run the clustering pipeline.
    """
    # 1. Load and Prepare Data
    try:
        file_path = 'barrettII_eyes_clustering.csv'
        original_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    print("Original Data Head:")
    print(original_data.head())
    
    # Keep track of original feature names
    features = original_data.columns.tolist()
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data)
    scaled_data_df = pd.DataFrame(scaled_data, columns=features)

    # 2. Determine the Optimal Number of Clusters
    optimal_k = find_optimal_k(scaled_data_df)

    # 3. Perform Final Clustering with Optimal k
    print(f"\n--- Performing final clustering with k={optimal_k} ---")
    kmeans_final = KMedoids(n_clusters=optimal_k, random_state=42, init='heuristic')
    labels = kmeans_final.fit_predict(scaled_data_df)
    
    # Add cluster labels to the original (unscaled) data for analysis
    original_data['Cluster'] = labels
    
    print("\nSize of each cluster:")
    print(original_data['Cluster'].value_counts().sort_index())
    
    # 4. Analyze and Visualize Results
    # Print centroids in the original scale for interpretability
    centroids_original_scale = scaler.inverse_transform(kmeans_final.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_original_scale, columns=features)
    print("\nCluster Centroids (in original data scale):")
    print(centroids_df.round(2))

    # Visualize the final clusters using PCA
    visualize_clusters_pca(scaled_data_df, labels, kmeans_final.cluster_centers_, features)
    
    # Display the first few rows of the data with assigned clusters
    print("\nData with assigned clusters:")
    print(original_data.head())

# %%
if __name__ == "__main__":
    main()
# %%

