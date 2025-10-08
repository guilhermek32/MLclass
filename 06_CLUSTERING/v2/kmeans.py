import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

# ============================
# 1. CARREGAR E PREPARAR DADOS
# ============================
file_path = 'RTVue_20221110_MLClass_cleaned.csv'
data_original = pd.read_csv(file_path)

print("Dataset original:")
print(data_original.head())
print(f"\nShape: {data_original.shape}")
print("="*100)

# ============================
# 2. PADRONIZAÇÃO DOS DADOS
# ============================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_original)
data_scaled_df = pd.DataFrame(data_scaled, columns=data_original.columns)

print("\nDados padronizados:")
print(data_scaled_df.head())
print("="*100)

# ============================
# 3. CONFIGURAÇÃO
# ============================
USE_KMEANS = True  # Mude para False se quiser usar K-Medoids
N_CLUSTERS = 16

# ============================
# 4. CLUSTERING COM K=10
# ============================
print(f"\nExecutando clustering com k={N_CLUSTERS}...")

if USE_KMEANS:
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
else:
    model = KMedoids(n_clusters=N_CLUSTERS, random_state=42, init='k-medoids++', 
                     max_iter=300, method='pam')

labels = model.fit_predict(data_scaled_df)
centroids = model.cluster_centers_

# Verificar clusters formados
n_unique_labels = len(np.unique(labels))
print(f"Clusters válidos formados: {n_unique_labels}")

if n_unique_labels > 1:
    sil_score = silhouette_score(data_scaled_df, labels)
    print(f"Silhouette Score: {sil_score:.4f}")
else:
    print("AVISO: Clustering falhou! Apenas 1 cluster formado.")
    sil_score = None

print("="*100)

# ============================
# 5. ADICIONAR LABELS AOS DADOS
# ============================
data_with_clusters = data_original.copy()
data_with_clusters['Cluster'] = labels

print("\nDados com clusters:")
print(data_with_clusters.head())
print("="*100)

# ============================
# 6. ANÁLISE DE DISTRIBUIÇÃO
# ============================
cluster_sizes = data_with_clusters['Cluster'].value_counts().sort_index()
print("\nTamanho de cada cluster:")
print(cluster_sizes)
print("\nDistribuição percentual:")
print((cluster_sizes / len(data_with_clusters) * 100).round(2))
print("="*100)

# ============================
# 7. CENTROIDES NO ESPAÇO ORIGINAL
# ============================
centroids_original_space = scaler.inverse_transform(centroids)
centroids_df = pd.DataFrame(centroids_original_space, columns=data_original.columns)

print("\nCentroides no espaço original:")
print(centroids_df)
print("="*100)

# ============================
# 8. AMOSTRAS DE CADA CLUSTER
# ============================
for cluster in range(N_CLUSTERS):
    n_samples = sum(data_with_clusters['Cluster'] == cluster)
    print(f"\nCluster {cluster} - Total de amostras: {n_samples}")
    if n_samples > 0:
        print(data_with_clusters[data_with_clusters['Cluster'] == cluster].head())
    else:
        print("  (cluster vazio)")
    print("-"*100)
print("="*100)

# ============================
# 9. VISUALIZAÇÃO: PCA 2D
# ============================
def visualize_clusters_pca(scaled_data, labels, centroids, n_clusters):
    """Visualiza clusters no espaço 2D usando PCA."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    reduced_centroids = pca.transform(centroids)
    
    explained_var = pca.explained_variance_ratio_
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=30, 
                         edgecolors='k', linewidth=0.3)
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
               c='red', marker='X', s=300, edgecolors='black', 
               linewidth=2, label='Centroides', zorder=10)
    
    algorithm = "K-Means" if USE_KMEANS else "K-Medoids"
    plt.title(f'Visualização dos Clusters com PCA ({algorithm}, k={n_clusters})', 
              fontsize=16, weight='bold')
    plt.xlabel(f'PCA 1 ({explained_var[0]*100:.1f}% variância)', fontsize=12)
    plt.ylabel(f'PCA 2 ({explained_var[1]*100:.1f}% variância)', fontsize=12)
    plt.legend(fontsize=12)
    plt.colorbar(scatter, label='Cluster', ticks=range(n_clusters))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clusters_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_clusters_pca(data_scaled_df, labels, centroids, N_CLUSTERS)
print("="*100)

# ============================
# 10. VISUALIZAÇÃO: SPIDER PLOTS
# ============================
def plot_spider_clusters_grid(centroids_df, cluster_sizes, ncols=4):
    """Cria um grid de spider plots, um para cada cluster."""
    anatomical_features = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
    available_features = [f for f in anatomical_features if f in centroids_df.columns]
    
    if len(available_features) == 0:
        print("AVISO: Nenhuma feature anatômica encontrada!")
        return
    
    centroids_anatomical = centroids_df[available_features]
    num_clusters = len(centroids_anatomical)
    nrows = (num_clusters + ncols - 1) // ncols
    
    num_vars = len(available_features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), 
                            subplot_kw=dict(polar=True))
    
    if num_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx in range(num_clusters):
        ax = axes[idx]
        values = centroids_anatomical.iloc[idx].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=f'C{idx%10}')
        ax.fill(angles, values, alpha=0.25, color=f'C{idx%10}')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_features, size=11, weight='bold')
        
        y_min = centroids_anatomical.min().min()
        y_max = centroids_anatomical.max().max()
        ax.set_ylim(y_min * 0.9, y_max * 1.1)
        
        n_samples = cluster_sizes.iloc[idx] if idx < len(cluster_sizes) else 0
        ax.set_title(f'Cluster {idx}\n(n={n_samples} amostras)', 
                     size=14, weight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
    
    for idx in range(num_clusters, len(axes)):
        fig.delaxes(axes[idx])
    
    algorithm = "K-Means" if USE_KMEANS else "K-Medoids"
    plt.suptitle(f'Perfis de Espessura Epitelial por Cluster ({algorithm})', 
                 fontsize=20, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('cluster_spider_plots_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_spider_clusters_grid(centroids_df, cluster_sizes, ncols=4)
print("="*100)

# ============================
# 11. RESUMO FINAL
# ============================
algorithm = "K-Means" if USE_KMEANS else "K-Medoids"
print("\n" + "="*100)
print(f"RESUMO DA ANÁLISE - {algorithm}")
print("="*100)
print(f"Número de clusters: {N_CLUSTERS}")
if sil_score:
    print(f"Silhouette Score: {sil_score:.4f}")
print(f"Total de amostras: {len(data_with_clusters)}")
print(f"\nDistribuição dos clusters:")
print(cluster_sizes.sort_values(ascending=False))
print("="*100)
