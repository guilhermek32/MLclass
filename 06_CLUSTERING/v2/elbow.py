import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


file = 'RTVue_20221110_MLClass_cleaned.csv'

data = pd.read_csv(file)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)



# Test k from 3 to 15
inertias = []
silhouette_scores = []
k_range = range(3, 16)




for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    
    inertias.append(kmeans.inertia_)  # Within-cluster sum of squares
    silhouette_scores.append(silhouette_score(data_scaled, labels))
    
# Plot both metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow plot
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12)
ax1.set_title('Elbow Method', fontsize=14)
ax1.grid(True, alpha=0.3)

# Silhouette plot
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axhline(y=0.5, color='r', linestyle='--', label='Reasonable threshold (0.5)')
ax2.axhline(y=0.7, color='orange', linestyle='--', label='Strong threshold (0.7)')
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Analysis', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

# Print scores
print("\nCluster Quality Metrics:")
print("k\tInertia\t\tSilhouette Score")
print("-" * 50)
for k, inertia, sil in zip(k_range, inertias, silhouette_scores):
    print(f"{k}\t{inertia:.2f}\t\t{sil:.4f}")


# For optimal k, analyze cluster profiles
optimal_k = 8  # Replace with your optimal k

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans_final.fit_predict(data_scaled)

# Calculate cluster centroids (average values)
cluster_profiles = data.groupby('Cluster').mean()
print("\nCluster Profiles (Mean Values):")
print(cluster_profiles)

# Calculate within-cluster variance
cluster_variance = data.groupby('Cluster').var()
print("\nCluster Variance:")
print(cluster_variance)

