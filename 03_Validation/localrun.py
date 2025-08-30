"""
Treino e avaliação (sem vazamento) no Abalone + PROGRESS BARS:
- FeatureEngineer + Winsorizer como transformers sklearn
- Pipeline com SMOTE, escala e PCA (whiten)
- Nested CV (outer 5x, inner 5x) para F1-macro
- Comparação de baselines (KNN, LogReg, HGB, RF)
- Barra de progresso para script geral, outer folds e dentro do GridSearch/CV (joblib)
- Compatível com Windows (joblib via threading)

Requisitos:
    pip install -U numpy pandas scikit-learn imbalanced-learn joblib tqdm tqdm-joblib
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict, Any

import numpy as np
import pandas as pd

# sklearn / imblearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, ParameterGrid
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# joblib threading backend para evitar erro _posixsubprocess no Windows
from joblib import parallel_backend

# tqdm (barras de progresso)
try:
    from tqdm.auto import tqdm
    from tqdm_joblib import tqdm_joblib
    HAS_TQDM = True
except Exception:
    # Fallback sem tqdm (script roda normal, só sem barras)
    HAS_TQDM = False
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)
    @contextmanager
    def tqdm_joblib(*args, **kwargs):
        yield

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Helper para paralelismo seguro (threads) + progresso joblib
# =========================
@contextmanager
def safe_threads():
    """Força joblib a usar 'threading' (em vez de processos)."""
    try:
        with parallel_backend('threading', n_jobs=1):
            yield
    except Exception as e:
        print(f"[warn] threading backend falhou: {e}\n-> caindo para execução sequencial")
        yield


@contextmanager
def joblib_progress(desc: str, total: int):
    """
    Contexto que combina threads + barra de progresso para tarefas joblib (GridSearchCV/cross_val_score).
    Se tqdm não estiver disponível, apenas executa normalmente.
    """
    if HAS_TQDM and total and total > 0:
        with safe_threads():
            with tqdm_joblib(tqdm(total=total, desc=desc, leave=False)):
                yield
    else:
        with safe_threads():
            yield


# =========================
# Utils
# =========================
def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return np.where(denom == 0, np.nan, numer / denom)


def count_grid_tasks(param_grid: Dict[str, Any], cv) -> int:
    """
    Estima o total de 'fits' do GridSearchCV = (#candidatos) * (#folds).
    """
    try:
        n_cands = len(list(ParameterGrid(param_grid)))
        n_folds = cv.get_n_splits() if hasattr(cv, "get_n_splits") else cv.n_splits
        return int(n_cands * n_folds)
    except Exception:
        return 0


# =========================
# Transformers customizados
# =========================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Parte do CSV bruto -> cria features e dummies de 'sex' sem vazamento.
    Mantém DataFrame com colunas numéricas finais + dummies sex_M/sex_I.
    """
    def __init__(self, drop_original: bool = True):
        self.drop_original = drop_original
        self._expected_cats = ["F", "M", "I"]  # ordem estável p/ dummies

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Verifica colunas necessárias
        req = {"sex", "length", "diameter", "height", "whole_weight",
               "shucked_weight", "viscera_weight", "shell_weight"}
        missing = req - set(X.columns)
        if missing:
            raise KeyError(f"Colunas ausentes no bruto: {sorted(missing)}")

        # Casting de sexo para categórico com categorias fixas
        X["sex"] = pd.Categorical(X["sex"], categories=self._expected_cats)

        # Novas features
        X["bmi"] = _safe_div(X["whole_weight"], X["height"] ** 2)
        X["length_dia_ratio"] = _safe_div(X["length"], X["diameter"])
        X["meat_yield"] = _safe_div(X["shucked_weight"], X["whole_weight"])
        X["shell_ratio"] = _safe_div(X["shell_weight"], X["whole_weight"])

        # Limpa infinities
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Dummies estáveis (F é baseline, gera sex_M e sex_I)
        dummies = pd.get_dummies(X["sex"], prefix="sex", drop_first=True)
        for col in ["sex_M", "sex_I"]:
            if col not in dummies.columns:
                dummies[col] = 0.0

        # Seleciona colunas finais
        keep = ["viscera_weight", "bmi", "length_dia_ratio", "meat_yield", "shell_ratio"]
        out = pd.concat([X[keep], dummies[["sex_M", "sex_I"]]], axis=1)

        # Drop de rows com NaN nas features geradas
        out = out.dropna(axis=0, how="any")

        return out


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsoriza por quantis, aprendidos no TRAIN apenas (sem vazamento).
    Funciona com DataFrame e preserva nomes de colunas.
    """
    def __init__(self, lower: float = 0.05, upper: float = 0.95):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None
        self.columns_: Optional[Iterable[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = X.columns
        q = X.quantile([self.lower, self.upper])
        self.lower_bounds_ = q.loc[self.lower]
        self.upper_bounds_ = q.loc[self.upper]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


# =========================
# Callable para peso customizado no KNN
# =========================
class PowerWeight:
    def __init__(self, alpha: float = 1.0, eps: float = 1e-9):
        self.alpha = alpha
        self.eps = eps

    def __call__(self, distances: np.ndarray) -> np.ndarray:
        return 1.0 / np.power(distances + self.eps, self.alpha)

    def __repr__(self):
        return f"PowerWeight(alpha={self.alpha})"


# =========================
# Construção de Pipeline
# =========================
def build_knn_pipeline(random_state: int = 42, use_smote: bool = True, use_pca: bool = True) -> ImbPipeline:
    steps = [
        ("fe", FeatureEngineer(drop_original=True)),
        ("winsor", Winsorizer(0.05, 0.95)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=random_state)))
    steps.extend([
        ("scaler", StandardScaler()),  # substituído via grid
    ])
    if use_pca:
        steps.append(("pca", PCA(whiten=True, random_state=random_state)))
    steps.append(("knn", KNeighborsClassifier()))
    return ImbPipeline(steps=steps)


def knn_param_grid(max_k: int = 61, step: int = 2, use_pca: bool = True, use_smote: bool = True) -> Dict[str, Any]:
    k_grid = list(range(3, max_k + 1, step))
    grid = {
        "scaler": [StandardScaler(), RobustScaler(), PowerTransformer(method="yeo-johnson")],
        "knn__n_neighbors": k_grid,
        "knn__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "knn__p": [1, 1.5, 2, 3],
        "knn__weights": ["uniform", "distance",
                         PowerWeight(alpha=0.5), PowerWeight(alpha=1.0), PowerWeight(alpha=2.0)],
    }
    if use_pca:
        grid["pca__n_components"] = [None, 0.95]
    if use_smote:
        grid["smote__k_neighbors"] = [3, 5, 7]
    return grid


# =========================
# Nested CV (outer imparcial, inner busca) com progresso
# =========================
def nested_cv_knn(
    X: pd.DataFrame, y: pd.Series,
    outer_splits: int = 5, inner_splits: int = 5, repeats: int = 1,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Retorna média e desvio de F1-macro no outer CV, com barras de progresso.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=repeats, random_state=random_state)
    scores = []

    total_outer = outer_cv.get_n_splits(X, y)
    outer_pbar = tqdm(total=total_outer, desc="Nested CV (outer folds)", leave=True) if HAS_TQDM else None

    fold = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_knn_pipeline(random_state=random_state, use_smote=True, use_pca=True)
        grid = knn_param_grid(max_k=61, step=2, use_pca=True, use_smote=True)

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=0
        )

        total_tasks = count_grid_tasks(grid, inner_cv)
        desc = f"Inner GridSearch fold {fold}/{total_outer}"
        with joblib_progress(desc, total_tasks):
            search.fit(X_tr, y_tr)

        y_pred = search.best_estimator_.predict(X_te)
        f1m = f1_score(y_te, y_pred, average="macro")
        balacc = balanced_accuracy_score(y_te, y_pred)

        scores.append(f1m)
        print(f"[Outer fold {fold}] best params: {search.best_params_}")
        print(f"[Outer fold {fold}] F1-macro: {f1m:.4f} | Balanced Acc: {balacc:.4f}")

        if outer_pbar:
            outer_pbar.update(1)

    if outer_pbar:
        outer_pbar.close()

    return float(np.mean(scores)), float(np.std(scores))


# =========================
# Baselines comparativos (mesmo pré-processamento) com progresso
# =========================
def build_common_pp(random_state: int = 42, use_smote: bool = True) -> ImbPipeline:
    steps = [
        ("fe", FeatureEngineer(drop_original=True)),
        ("winsor", Winsorizer(0.05, 0.95)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=random_state)))
    steps.append(("scaler", StandardScaler()))
    return ImbPipeline(steps=steps)


def compare_baselines(X: pd.DataFrame, y: pd.Series, cv_splits: int = 10, random_state: int = 42) -> None:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    pp = build_common_pp(random_state=random_state, use_smote=True)

    models = {
        "KNN (default)": KNeighborsClassifier(n_neighbors=31, p=1, weights="distance"),
        "LogReg (multinom, balanced)": LogisticRegression(
            multi_class="multinomial", class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "HGB (hist gradient boosting)": HistGradientBoostingClassifier(
            learning_rate=0.1, max_depth=None, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"
        ),
    }

    print("\n=== Baselines (F1-macro via CV) ===")
    for name, clf in models.items():
        pipe = ImbPipeline(pp.steps + [("clf", clf)])
        desc = f"CV {name}"
        total = cv.get_n_splits()
        with joblib_progress(desc, total):
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"{name:28s} -> mean={scores.mean():.4f}  std={scores.std():.4f}")


# =========================
# Execução principal (com progresso geral)
# =========================
def main():
    # Progresso geral (Nested CV 50% | Baselines 30% | Hold-out 20%)
    global_pbar = tqdm(total=100, desc="Progresso geral", leave=True) if HAS_TQDM else None

    # Carrega CSV BRUTO (com 'sex', 'length', ... e alvo 'type')
    csv_path = Path("abalone_dataset.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path.resolve()}")

    raw = pd.read_csv(csv_path)
    if "type" not in raw.columns:
        raise KeyError("Coluna-alvo 'type' não encontrada no CSV bruto.")

    y = raw["type"].astype(int)
    X = raw.drop(columns=["type"])

    # 1) Nested CV
    print("\n>>> Nested CV (outer=5x, inner=5x)")
    mean_f1, std_f1 = nested_cv_knn(X, y, outer_splits=5, inner_splits=5, repeats=1, random_state=42)
    print(f"\nNested CV F1-macro -> mean={mean_f1:.4f}  std={std_f1:.4f}")
    if global_pbar: global_pbar.update(50)

    # 2) Baselines
    compare_baselines(X, y, cv_splits=10, random_state=42)
    if global_pbar: global_pbar.update(30)

    # 3) Hold-out final (20%)
    print("\n>>> Hold-out final (20%)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_knn_pipeline(random_state=42, use_smote=True, use_pca=True)
    grid = knn_param_grid(max_k=61, step=2, use_pca=True, use_smote=True)

    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner_cv, n_jobs=-1, refit=True)

    total_tasks = count_grid_tasks(grid, inner_cv)
    with joblib_progress("GridSearch (hold-out)", total_tasks):
        search.fit(X_tr, y_tr)

    print("\nMelhores hiperparâmetros (hold-out):", search.best_params_)
    y_pred = search.best_estimator_.predict(X_te)
    print("\n--- Relatório no hold-out ---")
    print(classification_report(y_te, y_pred, digits=3))
    print("Matriz de confusão:\n", confusion_matrix(y_te, y_pred))

    if global_pbar:
        global_pbar.update(20)
        global_pbar.close()


if __name__ == "__main__":
    main()
 