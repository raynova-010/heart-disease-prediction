"""Feature definitions and row-wise cleaning shared by train.py and predict.py."""
import numpy as np
import pandas as pd

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

# Allowed values for each categorical feature (as they appear in the CSV).
CATEGORICAL = {
    "sex": ["Male", "Female"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": ["True", "False"],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": ["True", "False"],
    "slope": ["upsloping", "flat", "downsloping"],
    "thal": ["normal", "fixed defect", "reversable defect"],
}

FEATURES = NUMERIC + list(CATEGORICAL)

# Columns where 0 is physiologically impossible and means "not measured".
ZERO_AS_MISSING = ["chol", "trestbps"]


def clean(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise cleaning (safe to run inside the pipeline)."""
    X = X[FEATURES].copy()
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in ZERO_AS_MISSING:
        X.loc[X[col] == 0, col] = np.nan
    for col in CATEGORICAL:
        X[col] = X[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return X
