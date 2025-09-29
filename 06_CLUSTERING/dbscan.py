import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- 1. Carregar e preparar os dados ---
try:
    file_path = 'barrettII_eyes_clustering.csv'
    data = pd.read_csv(file_path)
    features = data.columns
    print("Arquivo CSV carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    print("Por favor, certifique-se de que o script está na mesma pasta que o arquivo CSV.")
    # Encerra o script se o arquivo não for encontrado
    exit()

# Lembre-se: DBSCAN é sensível à escala, então a padronização é crucial
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=features)

# --- 2. Encontrar o valor ideal para 'eps' ---
# Regra geral: min_samples = 2 * número de dimensões
min_samples = 2 * data_scaled.shape[1]  # 2 * 5 = 10
print(f"Usando min_samples = {min_samples}")

# Calcular a distância de cada ponto para seus vizinhos mais próximos
nn = NearestNeighbors(n_neighbors=min_samples)
nn.fit(data_scaled)
distances, indices = nn.kneighbors(data_scaled)

# Ordenar as distâncias do k-ésimo vizinho (k = min_samples)
k_distances = np.sort(distances[:, -1])

# Plotar o gráfico de distâncias para encontrar o "cotovelo"
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title('Gráfico de Distância K-vizinhos para encontrar o EPS ideal')
plt.xlabel('Pontos ordenados por distância')
plt.ylabel(f'Distância do {min_samples}º vizinho')
plt.grid(True)
plt.savefig('dbscan_eps_plot.png')
print("Gráfico para encontrar o 'eps' foi salvo como 'dbscan_eps_plot.png'.")



# --- 3. Executar o DBSCAN ---
# **AÇÃO NECESSÁRIA:** Analise o gráfico 'dbscan_eps_plot.png'
# Procure o "cotovelo" (o ponto de inflexão onde a curva sobe abruptamente)
# e defina o valor de 'eps_ideal' abaixo. Um bom chute inicial é 0.8
eps_ideal = 1.73  # <--- AJUSTE ESTE VALOR COM BASE NO GRÁFICO

dbscan = DBSCAN(eps=eps_ideal, min_samples=min_samples)
clusters = dbscan.fit_predict(data_scaled)

# Adicionar os resultados ao DataFrame original
data['Cluster_DBSCAN'] = clusters

# --- 4. Analisar os resultados ---
# O label -1 significa "ruído" (outlier)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"\n--- Resultados do DBSCAN ---")
print(f"Parâmetros: eps = {eps_ideal}, min_samples = {min_samples}")
print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de pontos de ruído (outliers): {n_noise}")
print("\nTamanho de cada cluster:")
print(data['Cluster_DBSCAN'].value_counts())

# --- 5. Visualizar os clusters ---
if n_clusters > 0:
    print("\nGerando pairplot para visualizar os clusters...")
    # O pairplot é ótimo para ver como os clusters se separam em diferentes dimensões
    pairplot = sns.pairplot(data, hue='Cluster_DBSCAN', palette='viridis', corner=True)
    pairplot.figure.suptitle("Clusters encontrados pelo DBSCAN", y=1.02)
    plt.savefig('dbscan_pairplot.png')
    print("Pairplot salvo como 'dbscan_pairplot.png'.")
else:
    print("\nNenhum cluster foi encontrado. Tente ajustar os parâmetros 'eps' e 'min_samples'.")
    
    
    
    
    

# --- 6. Analisar os Outliers (pontos com cluster -1) ---
outliers = data[data['Cluster_DBSCAN'] == -1]
padrao = data[data['Cluster_DBSCAN'] == 0]

print("\n--- Análise dos Olhos Atípicos (Outliers) ---")
print("Estatísticas dos Outliers:")
print(outliers.describe())

print("\n--- Análise dos Olhos Padrão ---")
print("Estatísticas do Cluster Padrão:")
print(padrao.describe())