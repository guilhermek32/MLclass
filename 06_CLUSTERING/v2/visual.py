import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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
USE_KMEANS = False  # Mude para False se quiser usar K-Medoids
N_CLUSTERS = 16

# ============================
# 4. CLUSTERING
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
                         c=labels, cmap='tab20', alpha=0.6, s=30, 
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
# 10. MAPEAMENTO ANATÔMICO DAS REGIÕES
# ============================
region_positions = {
    'C':  {'theta': 0,    'r': 0},      # Centro
    'S':  {'theta': 90,   'r': 1},      # Superior (12h)
    'ST': {'theta': 45,   'r': 1},      # Superior-Temporal (1h30)
    'T':  {'theta': 0,    'r': 1},      # Temporal (3h)
    'IT': {'theta': -45,  'r': 1},      # Inferior-Temporal (4h30)
    'I':  {'theta': -90,  'r': 1},      # Inferior (6h)
    'IN': {'theta': -135, 'r': 1},      # Inferior-Nasal (7h30)
    'N':  {'theta': 180,  'r': 1},      # Nasal (9h)
    'SN': {'theta': 135,  'r': 1}       # Superior-Nasal (10h30)
}

def polar_to_cartesian(theta_deg, r):
    """Converte coordenadas polares para cartesianas."""
    theta_rad = np.deg2rad(theta_deg)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    return x, y

# ============================
# 11. VISUALIZAÇÃO 3D: SUPERFÍCIES CORNEANAS
# ============================
def plot_3d_corneal_surfaces(centroids_df, cluster_sizes, ncols=6):
    """
    Cria um grid de superfícies 3D corneanas, uma para cada cluster.
    
    Args:
        centroids_df: DataFrame com centroides dos clusters
        cluster_sizes: Series com tamanho de cada cluster
        ncols: Número de colunas no grid
    """
    anatomical_features = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
    available_features = [f for f in anatomical_features if f in centroids_df.columns]
    
    if len(available_features) == 0:
        print("AVISO: Nenhuma feature anatômica encontrada!")
        return
    
    centroids_anatomical = centroids_df[available_features]
    num_clusters = len(centroids_anatomical)
    nrows = (num_clusters + ncols - 1) // ncols
    
    # Criar figura com subplots 3D
    fig = plt.figure(figsize=(7*ncols, 6*nrows))
    
    for idx in range(num_clusters):
        ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d')
        
        sample_data = centroids_anatomical.iloc[idx]
        
        # Preparar dados dos 9 pontos
        points_polar = []
        values = []
        
        for feature in available_features:
            theta = region_positions[feature]['theta']
            r = region_positions[feature]['r']
            x, y = polar_to_cartesian(theta, r)
            points_polar.append([x, y])
            values.append(sample_data[feature])
        
        points_polar = np.array(points_polar)
        values = np.array(values)
        
        # Criar grid para interpolação
        resolution = 50  # Resolução adequada para subplot
        xi = np.linspace(-1.3, 1.3, resolution)
        yi = np.linspace(-1.3, 1.3, resolution)
        XI, YI = np.meshgrid(xi, yi)
        
        # Interpolar valores para criar superfície suave
        ZI = griddata(points_polar, values, (XI, YI), method='cubic')
        
        # Aplicar máscara circular (apenas dentro da córnea)
        mask = XI**2 + YI**2 > 1.15**2
        ZI[mask] = np.nan
        
        # Determinar limites z baseado nos dados
        z_min = centroids_anatomical.min().min() * 0.9
        z_max = centroids_anatomical.max().max() * 1.1
        
        # Plotar superfície
        surf = ax.plot_surface(XI, YI, ZI, cmap='jet', 
                              linewidth=0, antialiased=True, 
                              alpha=0.9, vmin=z_min, vmax=z_max,
                              shade=True)
        
        # Adicionar pontos de medição originais
        points_x = points_polar[:, 0]
        points_y = points_polar[:, 1]
        ax.scatter(points_x, points_y, values, c='black', s=60, 
                  marker='o', edgecolors='white', linewidths=2, 
                  zorder=10)
        
        # Configurações dos eixos
        ax.set_xlabel('X', fontsize=9, weight='bold')
        ax.set_ylabel('Y', fontsize=9, weight='bold')
        ax.set_zlabel('μm', fontsize=9, weight='bold')
        ax.set_zlim(z_min, z_max)
        
        # Título
        n_samples = cluster_sizes.iloc[idx] if idx < len(cluster_sizes) else 0
        pct = (n_samples / cluster_sizes.sum() * 100) if cluster_sizes.sum() > 0 else 0
        ax.set_title(f'Cluster {idx}\n{n_samples} amostras ({pct:.1f}%)', 
                     size=12, weight='bold', pad=15)
        
        # Ajustar visualização
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        
        # Remover ticks para clareza
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
    
    algorithm = "K-Means" if USE_KMEANS else "K-Medoids"
    plt.suptitle(f'Superfícies 3D de Espessura Epitelial por Cluster ({algorithm})', 
                 fontsize=20, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('corneal_3d_surfaces_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nGerando visualizações 3D das superfícies corneanas...")
plot_3d_corneal_surfaces(centroids_df, cluster_sizes, ncols=4)
print("✓ Superfícies 3D salvas em 'corneal_3d_surfaces_grid.png'")
print("="*100)

# ============================
# 12. VISUALIZAÇÃO 3D: AMOSTRAS INDIVIDUAIS (OPCIONAL)
# ============================
def plot_3d_individual_samples(data_with_clusters, n_samples_per_cluster=2):
    """
    Plota amostras individuais de cada cluster em 3D.
    
    Args:
        data_with_clusters: DataFrame com dados e cluster labels
        n_samples_per_cluster: Número de amostras a plotar por cluster
    """
    anatomical_features = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
    
    # Selecionar amostras
    np.random.seed(42)
    sample_list = []
    
    for cluster_id in range(min(3, N_CLUSTERS)):  # Apenas primeiros 3 clusters
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
        if len(cluster_data) >= n_samples_per_cluster:
            samples = cluster_data.sample(n_samples_per_cluster)
            for idx, row in samples.iterrows():
                sample_list.append((idx, cluster_id, row))
    
    if len(sample_list) == 0:
        print("Não há amostras suficientes para plotar")
        return
    
    # Criar grid
    n_plots = len(sample_list)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(18, 6*nrows))
    
    for plot_idx, (sample_idx, cluster_id, sample_data) in enumerate(sample_list):
        ax = fig.add_subplot(nrows, ncols, plot_idx+1, projection='3d')
        
        # Preparar dados
        points_polar = []
        values = []
        
        for feature in anatomical_features:
            theta = region_positions[feature]['theta']
            r = region_positions[feature]['r']
            x, y = polar_to_cartesian(theta, r)
            points_polar.append([x, y])
            values.append(sample_data[feature])
        
        points_polar = np.array(points_polar)
        values = np.array(values)
        
        # Grid
        resolution = 60
        xi = np.linspace(-1.3, 1.3, resolution)
        yi = np.linspace(-1.3, 1.3, resolution)
        XI, YI = np.meshgrid(xi, yi)
        
        # Interpolação
        ZI = griddata(points_polar, values, (XI, YI), method='cubic')
        mask = XI**2 + YI**2 > 1.15**2
        ZI[mask] = np.nan
        
        # Plot
        surf = ax.plot_surface(XI, YI, ZI, cmap='jet', 
                              linewidth=0, antialiased=True, 
                              alpha=0.9, vmin=40, vmax=70,
                              shade=True)
        
        points_x = points_polar[:, 0]
        points_y = points_polar[:, 1]
        ax.scatter(points_x, points_y, values, c='black', s=80, 
                  marker='o', edgecolors='white', linewidths=2, zorder=10)
        
        ax.set_xlabel('X', fontsize=10, weight='bold')
        ax.set_ylabel('Y', fontsize=10, weight='bold')
        ax.set_zlabel('μm', fontsize=10, weight='bold')
        ax.set_zlim(35, 75)
        ax.set_title(f'Amostra {sample_idx}\nCluster {cluster_id}', 
                    fontsize=12, weight='bold', pad=15)
        ax.view_init(elev=25, azim=45)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Exemplos de Superfícies 3D Individuais', 
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('corneal_3d_individual_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nGerando exemplos de amostras individuais em 3D...")
plot_3d_individual_samples(data_with_clusters, n_samples_per_cluster=2)
print("✓ Amostras individuais salvas em 'corneal_3d_individual_samples.png'")
print("="*100)

# ============================
# 13. RESUMO FINAL
# ============================
algorithm = "K-Means" if USE_KMEANS else "K-Medoids"
print("\n" + "="*100)
print(f"RESUMO DA ANÁLISE - {algorithm}")
print("="*100)
print(f"Número de clusters: {N_CLUSTERS}")
print(f"Clusters válidos formados: {n_unique_labels}")
print("\nTamanho de cada cluster:")
print(cluster_sizes.sort_values(ascending=False))
