import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print('\n - Lendo o arquivo com o dataset sobre diabetes')
# Altere 'seu_arquivo.csv' para o nome real do seu arquivo
data = pd.read_csv('preprocessed_dataset.csv') 

# --- PASSO DE VERIFICAÇÃO ---
# Adicione esta linha para ver os nomes exatos das colunas
print("\nColunas encontradas no arquivo CSV:")
print(data.columns)
# -----------------------------

# Criando X and y para o algoritmo de aprendizagem
print(' - Separando features (X) e target (y)')

# Verifique se os nomes abaixo estão EXATAMENTE IGUAIS aos impressos acima
feature_cols = ['Pregnancies', 'Glucose', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', 'Age']

# O erro acontece aqui se os nomes não baterem
X = data[feature_cols] 
y = data.Outcome

# O restante do código continua igual...
print(' - Dividindo os dados em Treino e Teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f' - Dimensões dos dados -> Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras')

print(' - Criando e treinando o modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

print(' - Realizando previsões nos dados de teste para avaliação')
y_pred = neigh.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n----------- RESULTADO -----------")
print(f"A acurácia do modelo é: {accuracy * 100:.2f}%")
print("---------------------------------")