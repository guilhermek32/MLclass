import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        
def winsorize_dataframe(df, cols, lower_percentile=0.05, upper_percentile=0.95):
    """
    Aplica a Winsorização nas colunas especificadas de um DataFrame.

    Parameters:
    - df: DataFrame com os dados.
    - cols: Lista de colunas a serem Winsorizadas.
    - lower_percentile: Percentil inferior para truncamento (padrão: 5%).
    - upper_percentile: Percentil superior para truncamento (padrão: 95%).

    Returns:
    - DataFrame com as colunas Winsorizadas.
    """
    # Loop sobre as colunas a serem Winsorizadas
    for col in cols:
        # Calcular os percentis inferior e superior
        lower_limit = df[col].quantile(lower_percentile)
        upper_limit = df[col].quantile(upper_percentile)

        # Aplicar a Winsorização: substituir valores abaixo do limite inferior ou acima do limite superior
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    return df


import pandas as pd

def max_normalize(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Retorna uma nova cópia de df em que cada coluna é dividida pelo seu valor máximo,
    escalando os dados para [0, 1].

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame original com as features a normalizar.
    cols : list[str] ou None, opcional
        Lista de nomes de colunas a normalizar. Se None, normaliza todas as colunas numéricas.

    Retorno
    -------
    pd.DataFrame
        DataFrame normalizado pelo valor máximo.
    """
    # cópia para não alterar o original
    df_norm = df.copy()
    
    # seleciona colunas numéricas, se não especificadas
    if cols is None:
        cols = df_norm.select_dtypes(include="number").columns.tolist()
    
    # calcula máximos e faz a normalização
    max_vals = df_norm[cols].max()
    # evita divisão por zero
    max_vals = max_vals.replace({0: 1})
    
    # divisão vetorizada por coluna
    df_norm[cols] = df_norm[cols].div(max_vals)
    
    return df_norm

    
        
if __name__ == "__main__":
    
    input_file = 'diabetes_dataset.csv'

    # 1) Carrega dados
    df = load_data(input_file)
    if df is None:
        exit(1)
        
        
    # 2) Informação inicial
    print("Antes do pré-processamento:")
    df.info()
   
    # 3) Pré-processamento
    print("\n - Iniciando pré-processamento dos dados")
    
    #detect when there are more than two missing values in one line
    print("\n - Verificando colunas e removendo colunas desnecessárias")
    # Detectando linhas com mais de dois valores ausentes
    rows_with_missing = df.isnull().sum(axis=1)
    rows_to_drop = rows_with_missing[rows_with_missing > 2].index
    df = df.drop(index=rows_to_drop)
    print(f"Removidas {len(rows_to_drop)} linhas com mais de dois valores ausentes.")
    #4) drop when there are more than two outliers, DO work!! 
    print("Linhas com valores ausentes removidas!")
    
   
    print(" - Verificando colunas e removendo colunas desnecessárias")
    df = df.drop(columns=['Insulin'], errors='ignore')  # Removendo a coluna Insulin
    #df = df.drop(columns=['BloodPressure'], errors='ignore')  # Removendo a coluna BloodPressure
    #df = df.drop(columns=['Pregnancies'], errors='ignore')  # Removendo a coluna Pregnancies
    #df = df.drop(columns=['DiabetesPedigreeFunction'], errors='ignore')  # Removendo a coluna SkinThickness
    
    mean_values = df.mean()
    
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())  # Preenchendo valores ausentes de BMI com a mediana
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())  # Preenchendo valores ausentes de SkinThickness com a mediana
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())  # Preenchendo valores ausentes de Glucose com a mediana
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median())  # Preenchendo valores ausentes de BloodPressure com a mediana
    
    
    # 5) Age grouping do not work
    
    
    
    #Normalização dos dados
    #print(" - Normalizando os dados antes de Windsorização")
    #df = max_normalize(df, cols=df.columns)
    #scaler = MinMaxScaler()
    #df[df.columns] = scaler.fit_transform(df[df.columns])
    #df = max_normalize(df, cols=df.columns)
    
    # Winsorização
    print(" - Aplicando Winsorização para tratar outliers")
    cols_to_winsorize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df = winsorize_dataframe(df, cols_to_winsorize)
    
    #Normalização dos dados
    print(" - Normalizando os dados após Winsorização")
    df = max_normalize(df, cols=df.columns)
    
    
    df.info()
    
    df = save_data(df, 'preprocessed_dataset.csv')