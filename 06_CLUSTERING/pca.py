import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Carregar e padronizar os dados
file_path = 'barrettII_eyes_clustering.csv'
data = pd.read_csv(file_path)
features = data.columns

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=features)

# 2. Re-executar o K-Means para obter os rótulos (labels) dos clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(data_scaled)

# 3. Aplicar o PCA para reduzir para 2 dimensões
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# 4. Visualizar os clusters no gráfico PCA
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.8)
plt.title('Visualização dos Clusters com PCA', fontsize=16)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.grid(True)
plt.legend(title='Cluster')
plt.savefig('pca_clusters.png')
plt.show()

# 5. Analisar os Componentes Principais
print("--- Análise dos Componentes Principais ---")
print(f"Variância explicada por cada componente: {pca.explained_variance_ratio_}")
print(f"Variância total explicada pelos 2 componentes: {sum(pca.explained_variance_ratio_):.2%}")

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
print("\nImportância de cada variável para os componentes:")
print(loadings)