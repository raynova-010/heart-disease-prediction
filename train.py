"""Train, tune and compare heart disease classifiers.

Usage: .venv/bin/python train.py
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from heart_features import CATEGORICAL, NUMERIC, clean

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "heart_disease_uci.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42


def load_data():
    df = pd.read_csv(DATA)
    y = (df["num"] > 0).astype(int)  # any disease severity -> 1
    X = df.drop(columns=["id", "dataset", "num"])
    return X, y


def make_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, list(CATEGORICAL)),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean)),
        ("preprocess", preprocess),
        ("model", model),
    ])


CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, random_state=SEED),
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3, 5],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=SEED),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    ),
}


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    X, y = load_data()
    # Split first; every fitted step lives in the pipeline, so CV and test are leak-free.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
          f"positive rate train={y_train.mean():.3f} test={y_test.mean():.3f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted, matrices = [], {}, {}

    for name, (model, grid) in CANDIDATES.items():
        search = GridSearchCV(
            make_pipeline(model), grid, cv=cv, n_jobs=-1,
            scoring={"recall": "recall", "f1": "f1"}, refit="recall",
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        cv_f1 = search.cv_results_["mean_test_f1"][search.best_index_]
        print(f"{name}: best CV recall={search.best_score_:.3f} (F1={cv_f1:.3f})")
        print(f"  params: { {k.removeprefix('model__'): v for k, v in search.best_params_.items()} }")

        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "cv_recall": search.best_score_,
            "cv_f1": cv_f1,
            "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        })
        fitted[name] = best
        matrices[name] = confusion_matrix(y_test, pred)

    table = (pd.DataFrame(rows)
             .sort_values(["recall", "f1"], ascending=False)
             .reset_index(drop=True))
    table.to_csv(REPORTS / "comparison.csv", index=False)

    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5))
    for ax, (name, cm) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name)
    fig.suptitle("Test set confusion matrices (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrices.png", dpi=150)

    best_name = table.loc[0, "model"]
    joblib.dump(fitted[best_name], MODELS / "best_model.joblib")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print("\nTest set results:")
        print(table.drop(columns="best_params").to_string(index=False))
    for name, cm in matrices.items():
        tn, fp, fn, tp = cm.ravel()
        print(f"{name:20s} TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"\nBest model (recall, then F1): {best_name} -> {MODELS / 'best_model.joblib'}")
    print(f"Saved {REPORTS / 'comparison.csv'} and {REPORTS / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
