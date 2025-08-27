import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint

# This function will contain the main logic of your script.
def run_abalone_experiment():
    # 1) Carregar dados ------------------------------------------------------------
    CSV_PATH = "abalone_app.csv"
    df = pd.read_csv(CSV_PATH)

    assert "type" in df.columns, "Coluna 'type' não encontrada no CSV."

    # 2) Feature Engineering ----------------------------------------
    print("Criando novas features...")
    df['bmi'] = df['whole_weight'] / (df['height'] ** 2)
    df['length_dia_ratio'] = df['length'] / df['diameter']
    df['meat_yield'] = df['shucked_weight'] / df['whole_weight']
    df['shell_ratio'] = df['shell_weight'] / df['whole_weight']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Novas features criadas: bmi, length_dia_ratio, meat_yield, shell_ratio")

    # 3) Definir X e y -------------------------------------------------------------
    y = df["type"].astype(str)

    num_cols = [
        "length", "diameter", "height", "whole_weight", "shucked_weight",
        "viscera_weight", "shell_weight", "bmi", "length_dia_ratio",
        "meat_yield", "shell_ratio"
    ]
    cat_cols = ["sex"]
    X = df[cat_cols + num_cols].copy()

    # 4) Split estratificado -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\nDados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # 5) Pré-processamento + Modelo ------------------------------------------------
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

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("rf", rf)
    ])

    # 6) Hyperparameter Tuning ----------------------------------------
    param_dist = {
        'rf__n_estimators': randint(100, 1000),
        'rf__max_depth': [None, 10, 20, 30, 40, 50],
        'rf__min_samples_split': randint(2, 20),
        'rf__min_samples_leaf': randint(1, 20),
        'rf__max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=1,
        scoring='accuracy'
    )

    print("\nIniciando busca de hiperparâmetros com RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    # 7) Avaliar o melhor modelo encontrado ----------------------------------------
    print("\nBusca concluída. Melhores parâmetros encontrados:")
    print(random_search.best_params_)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification report (macro) com o modelo otimizado:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = np.mean(y_pred == y_test)
    print(f"\nAcurácia final no conjunto de teste: {accuracy:.4f}\n")

    # 8) (Opcional) Importâncias de atributos ---------------------
    try:
        feature_names = best_model.named_steps["prep"].get_feature_names_out()
        importances = best_model.named_steps["rf"].feature_importances_
        idx = np.argsort(importances)[::-1]
        top = 15
        print("\nTop importâncias de atributos do modelo final:")
        for i in idx[:top]:
            print(f"{feature_names[i]:<40s} {importances[i]:.4f}")
    except Exception as e:
        print("\n(Não foi possível listar importâncias de atributos)", e)


# This is the crucial part for Windows multiprocessing
if __name__ == "__main__":
    run_abalone_experiment()