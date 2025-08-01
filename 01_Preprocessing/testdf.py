import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Carrega dados de um CSV e retorna um DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def save_data(data, file_path): 
    """Salva um DataFrame em CSV."""
    try:
        data.to_csv(file_path, index=False)
        print(f"Dados salvos em {file_path}")
    except Exception as e:
        print(f"Erro ao salvar dados: {e}")
        
        

        
        
if __name__ == "__main__":
    input_file  = 'preprocessed_dataset.csv'

    # 1) Carrega dados
    df = load_data(input_file)
    if df is None:
        exit(1)
        
    # Gerar gráfico de bloxplot para todos os dados
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.title('Boxplot dos Dados Originais')
    plt.tight_layout()
    plt.show()

    correlacao = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor da Correlação entre Variáveis')
    plt.show()

    # 2) Informação inicial
    print("Antes do pré-processamento:")
    print(df.info())
