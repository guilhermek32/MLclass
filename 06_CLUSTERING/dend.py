import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Carregar e padronizar os dados
file_path = 'barrettII_eyes_clustering.csv'
data = pd.read_csv(file_path)
features = data.columns

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=features)

# 2. Gerar a matriz de ligação para o dendrograma
# O método 'ward' minimiza a variância dentro dos clusters que estão sendo mesclados
linked = linkage(data_scaled, method='ward')

# 3. Plotar o dendrograma
plt.figure(figsize=(16, 8))
plt.title('Dendrograma do Clustering Hierárquico', fontsize=16)
plt.xlabel('Tamanho do Cluster (ou índice da amostra)', fontsize=12)
plt.ylabel('Distância de Ward (dissimilaridade)', fontsize=12)
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           truncate_mode='lastp',  # Mostra os últimos 'p' clusters mesclados
           p=12,  # Apenas para deixar o gráfico mais legível
           show_leaf_counts=True,
           show_contracted=True)
plt.savefig('hierarchical_dendrogram.png')
plt.show()

print("Dendrograma salvo como 'hierarchical_dendrogram.png'")