# All necessary imports, including 'requests'
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint

def run_abalone_experiment():
    # 1) Carregar dados de TREINO --------------------------------------------------
    TRAIN_CSV_PATH = "abalone_dataset.csv" # Or preprocessed_dataset.csv
    df_train = pd.read_csv(TRAIN_CSV_PATH)

    # 2) Feature Engineering (para os dados de treino) -----------------------------
    print("Criando novas features para o dataset de treino...")
    df_train['bmi'] = df_train['whole_weight'] / (df_train['height'] ** 2)
    df_train['length_dia_ratio'] = df_train['length'] / df_train['diameter']
    df_train['meat_yield'] = df_train['shucked_weight'] / df_train['whole_weight']
    df_train['shell_ratio'] = df_train['shell_weight'] / df_train['whole_weight']
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3) Definir X e y para o TREINO -----------------------------------------------
    y_train_full = df_train["type"].astype(str)
    num_cols = [
        "length", "diameter", "height", "whole_weight", "shucked_weight",
        "viscera_weight", "shell_weight", "bmi", "length_dia_ratio",
        "meat_yield", "shell_ratio"
    ]
    cat_cols = ["sex"]
    X_train_full = df_train[cat_cols + num_cols].copy()

    # 4) Pré-processamento e Definição do Modelo ------------------------------------
    # (Note: We are not splitting the data anymore, we train on the full dataset)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    # Usando os melhores parâmetros que você encontrou
    print("Usando os melhores parâmetros encontrados na sua busca.")
    best_params_from_search = {
        'n_estimators': 171,
        'min_samples_split': 12,
        'min_samples_leaf': 15,
        'max_features': 'sqrt',
        'max_depth': 30
    }

    rf_optimized = RandomForestClassifier(
        **best_params_from_search,
        random_state=42,
        class_weight="balanced_subsample"
    )
    
    # Criar o pipeline final com o modelo otimizado
    final_model = Pipeline(steps=[
        ("prep", preprocess),
        ("rf", rf_optimized)
    ])

    # 5) Treinar o modelo final com TODOS os dados de treino -----------------------
    print("\nTreinando o modelo final com o dataset completo...")
    final_model.fit(X_train_full, y_train_full)
    print("Modelo treinado com sucesso!")

    # 6) Carregar os dados de APLICAÇÃO (para enviar ao servidor) -----------------
    print("\nCarregando 'abalone_app.csv' para fazer as previsões...")
    data_app = pd.read_csv('abalone_app.csv')

    # 7) APLICAR A MESMA FEATURE ENGINEERING -------------------------------------
    print("Aplicando a mesma feature engineering nos dados de 'app'...")
    data_app['bmi'] = data_app['whole_weight'] / (data_app['height'] ** 2)
    data_app['length_dia_ratio'] = data_app['length'] / data_app['diameter']
    data_app['meat_yield'] = data_app['shucked_weight'] / data_app['whole_weight']
    data_app['shell_ratio'] = data_app['shell_weight'] / data_app['whole_weight']
    data_app.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Garantir que as colunas estão na mesma ordem
    X_app = data_app[cat_cols + num_cols]

    # 8) Fazer as previsões nos dados de aplicação --------------------------------
    # O pipeline 'final_model' fará todo o pré-processamento automaticamente
    print("Fazendo previsões com o modelo treinado...")
    y_pred_app = final_model.predict(X_app)

    # 9) Enviar as previsões para o servidor --------------------------------------
    print("Enviando previsões para o servidor...")
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"
    
    # TODO: Substitua pela sua chave aqui
    DEV_KEY = "Trio Ternura"

    # Criar o JSON para ser enviado
    data_to_send = {
        'dev_key': DEV_KEY,
        'predictions': pd.Series(y_pred_app).to_json(orient='values')
    }

    # Enviar a requisição POST e salvar a resposta
    r = requests.post(url=URL, data=data_to_send)

    # Extrair e imprimir o texto da resposta do servidor
    print("\n- Resposta do servidor:\n", r.text, "\n")


# Ponto de entrada do script
if __name__ == "__main__":
    run_abalone_experiment()