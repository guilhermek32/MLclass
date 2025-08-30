import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Utilidades de I/O
# =========================
def load_data(file_path: str) -> pd.DataFrame | None:
    """Carrega dados de um CSV e retorna um DataFrame, ou None em caso de erro."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Salva um DataFrame em CSV (sem sobrescrever o df chamador)."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Dados salvos em {file_path}")
    except Exception as e:
        print(f"Erro ao salvar dados: {e}")


# =========================
# Pré-processamento
# =========================
def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aplica One-Hot Encoding nas colunas categóricas especificadas (se existirem)."""
    presentes = [c for c in cols if c in df.columns]
    if not presentes:
        return df
    return pd.get_dummies(df, columns=presentes, drop_first=True)


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Divisão segura: evita divisão por zero retornando NaN nesse caso."""
    return np.where(denom == 0, np.nan, numer / denom)


def add_features(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    """
    Cria features:
      - bmi = whole_weight / height^2
      - length_dia_ratio = length / diameter
      - meat_yield = shucked_weight / whole_weight
      - shell_ratio = shell_weight / whole_weight
    """
    X = df.copy()

    # Garantir colunas necessárias
    req = {"whole_weight", "height", "length", "diameter", "shucked_weight", "shell_weight"}
    missing = req - set(X.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para feature engineering: {sorted(missing)}")

    X["bmi"] = _safe_div(X["whole_weight"], X["height"] ** 2)
    X["length_dia_ratio"] = _safe_div(X["length"], X["diameter"])
    X["meat_yield"] = _safe_div(X["shucked_weight"], X["whole_weight"])
    X["shell_ratio"] = _safe_div(X["shell_weight"], X["whole_weight"])

    # Limpa infinitos/NaN criados pelas divisões
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(subset=["bmi", "length_dia_ratio", "meat_yield", "shell_ratio"], inplace=True)

    if drop_original:
        X.drop(columns=["length", "diameter", "height", "whole_weight", "shucked_weight", "shell_weight"], inplace=True)

    return X


def winsorize_dataframe(df: pd.DataFrame, cols: list[str],
                        lower_percentile: float = 0.05, upper_percentile: float = 0.95) -> pd.DataFrame:
    """
    Aplica winsorização vetorizada com clip por coluna.
    """
    X = df.copy()
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return X

    q = X[cols].quantile([lower_percentile, upper_percentile])
    lower = q.loc[lower_percentile]
    upper = q.loc[upper_percentile]

    # clip com broadcast por coluna
    X[cols] = X[cols].clip(lower=lower, upper=upper, axis=1)
    return X


def max_normalize(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Normaliza para [0,1] dividindo cada coluna pelo seu máximo (evita dividir por zero).
    """
    X = df.copy()
    if cols is None:
        cols = X.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return X

    max_vals = X[cols].max().replace(0, 1)  # evita divisão por zero
    X[cols] = X[cols].div(max_vals)
    return X


# =========================
# Visualização
# =========================
def plot_boxplot(df: pd.DataFrame, cols: list[str], title: str) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("Nenhuma coluna numérica válida para plotar.")
        return
    plt.figure(figsize=(12, 6))
    df[cols].boxplot()
    plt.title(title)
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_max_stats(df: pd.DataFrame, cols: list[str], header: str) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    print(header)
    print("Máximos por coluna:")
    print(df[cols].max())
    print("\nLinhas onde cada coluna atinge o máximo:")
    print(df.loc[df[cols].idxmax()])


# =========================
# Execução principal
# =========================
def main() -> None:
    input_file = "abalone_dataset.csv"
    output_file = "preprocessed_dataset.csv"

    # 1) Carrega dados
    df = load_data(input_file)
    if df is None:
        raise SystemExit(1)

    print("Antes do pré-processamento:")
    df.info()

    # Drop de índice específico (se existir)
    if 2134 in df.index:
        print("Removendo linha com índice 2134.")
        df = df.drop(index=2134)

    # 2) One-Hot e Features
    df = one_hot_encode(df, cols=["sex"])
    df = add_features(df)

    # Colunas alvo para tratamento
    num_cols = ["viscera_weight", "bmi", "length_dia_ratio", "meat_yield", "shell_ratio"]
    num_cols = [c for c in num_cols if c in df.columns]

    # 3) Diagnóstico antes de winsorizar/normalizar
    print_max_stats(df, num_cols, header="\n=== Estatísticas antes da winsorização ===")
    plot_boxplot(df, num_cols, "Boxplot (antes da winsorização)")

    # 4) Winsorização + Normalização
    df = winsorize_dataframe(df, num_cols, 0.05, 0.95)
    df = max_normalize(df, num_cols)

    # 5) Diagnóstico depois
    print_max_stats(df, num_cols, header="\n=== Estatísticas após winsorização + normalização ===")
    plot_boxplot(df, num_cols, "Boxplot (após winsorização + normalização)")

    print("\nApós o pré-processamento:")
    df.info()

    # 6) Salva resultado
    save_data(df, output_file)


if __name__ == "__main__":
    main()
