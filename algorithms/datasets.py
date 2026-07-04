"""Dataset generation, real/Kaggle dataset loading, and CSV upload handling.

All tabular loaders return (X, y, feature_names, messages): X keeps EVERY
selected feature (so models train on the full feature set); plotting decides
separately how to project to 1-D/2-D."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets as skd
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- synthetic

@st.cache_data
def make_synthetic(kind: str, n_samples: int, noise: float):
    """Generate a synthetic dataset. `noise` is a 0..1 knob mapped to each
    generator's natural noise scale."""
    rng = np.random.RandomState(RNG_SEED)

    # --- 1-D regression curves (X is (n, 1)) ---
    if kind == "Noisy sine":
        X = rng.uniform(-3, 3, n_samples).reshape(-1, 1)
        y = np.sin(X[:, 0] * 1.5) + rng.normal(0, 0.1 + noise, n_samples)
        return X, y, ["x"]
    if kind == "Linear":
        X = rng.uniform(-3, 3, n_samples).reshape(-1, 1)
        y = 0.8 * X[:, 0] + 1.0 + rng.normal(0, 0.2 + 2 * noise, n_samples)
        return X, y, ["x"]
    if kind == "Cubic":
        X = rng.uniform(-2, 2, n_samples).reshape(-1, 1)
        y = X[:, 0] ** 3 - 2 * X[:, 0] + rng.normal(0, 0.2 + 2 * noise, n_samples)
        return X, y, ["x"]

    # --- 2-D classification / clustering ---
    names = ["feature 1", "feature 2"]
    if kind == "Moons":
        X, y = skd.make_moons(n_samples=n_samples, noise=0.05 + 0.3 * noise,
                              random_state=RNG_SEED)
    elif kind == "Circles":
        X, y = skd.make_circles(n_samples=n_samples, noise=0.02 + 0.2 * noise,
                                factor=0.5, random_state=RNG_SEED)
    elif kind == "Blobs":
        X, y = skd.make_blobs(n_samples=n_samples, centers=3,
                              cluster_std=0.6 + 2.0 * noise, random_state=RNG_SEED)
    elif kind == "Anisotropic blobs":
        X, y = skd.make_blobs(n_samples=n_samples, centers=3,
                              cluster_std=0.6 + 1.5 * noise, random_state=RNG_SEED)
        X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])  # shear the blobs
    elif kind == "Two informative features":
        X, y = skd.make_classification(
            n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
            n_clusters_per_class=1, flip_y=0.02 + 0.3 * noise,
            class_sep=1.2, random_state=RNG_SEED)
    elif kind == "Spirals":  # two intertwined spirals - hard for linear models
        n = n_samples // 2
        t = np.sqrt(rng.rand(n)) * 3 * np.pi
        s1 = np.c_[t * np.cos(t), t * np.sin(t)]
        s2 = np.c_[t * np.cos(t + np.pi), t * np.sin(t + np.pi)]
        X = np.vstack([s1, s2]) / 3 + rng.normal(0, 0.05 + 0.4 * noise, (2 * n, 2))
        y = np.r_[np.zeros(n, int), np.ones(n, int)]
    else:
        raise ValueError(f"Unknown synthetic dataset: {kind}")
    return X, y, names


# ---------------------------------------------------------------- real data

_SKLEARN_LOADERS = {
    "Iris": skd.load_iris,
    "Wine": skd.load_wine,
    "Breast Cancer": skd.load_breast_cancer,
    "Diabetes": skd.load_diabetes,
}

# bundled Kaggle classics: (filename, target column, feature columns)
KAGGLE_SPECS = {
    "Titanic (Kaggle)": ("titanic.csv", "survived",
                         ["pclass", "sex", "age", "sibsp", "parch", "fare",
                          "embarked"]),
    "Penguins (Kaggle)": ("penguins.csv", "species",
                          ["island", "bill_length_mm", "bill_depth_mm",
                           "flipper_length_mm", "body_mass_g", "sex"]),
    "Tips (Kaggle)": ("tips.csv", "tip",
                      ["total_bill", "sex", "smoker", "day", "time", "size"]),
    "Auto MPG (Kaggle)": ("mpg.csv", "mpg",
                          ["cylinders", "displacement", "horsepower", "weight",
                           "acceleration", "model_year", "origin"]),
    "Pima Diabetes (Kaggle)": ("pima_diabetes.csv", "Outcome",
                               ["Pregnancies", "Glucose", "BloodPressure",
                                "SkinThickness", "Insulin", "BMI",
                                "DiabetesPedigreeFunction", "Age"]),
    "Diamonds (Kaggle)": ("diamonds.csv", "price",
                          ["carat", "cut", "color", "clarity", "depth", "table"]),
}


@st.cache_data
def load_real(name: str):
    """Load an sklearn classic with ALL features, standardized."""
    data = _SKLEARN_LOADERS[name]()
    X = StandardScaler().fit_transform(data.data)
    return X, data.target, list(data.feature_names), []


@st.cache_data
def load_kaggle(name: str, task: str):
    """Load a bundled Kaggle-classic CSV through the same cleaning pipeline as
    user uploads (one-hot encoding, imputation, ...)."""
    fname, target, features = KAGGLE_SPECS[name]
    df = pd.read_csv(DATA_DIR / fname)
    return prepare_csv(df, features, None if task == "clustering" else target, task)


# ---------------------------------------------------------------- CSV upload

MAX_CATEGORIES = 15  # one-hot limit; beyond this a text column is likely an ID


def prepare_csv(df: pd.DataFrame, feature_cols, target_col, task):
    """Clean a tabular dataset: median-impute numeric NaNs, one-hot encode
    low-cardinality categorical features, drop high-cardinality text columns,
    encode a non-numeric target. Keeps ALL resulting features (standardized).
    Returns (X, y, feature_names, messages). `target_col` may be None."""
    msgs = []
    numeric, categorical, dropped = [], [], []
    # a text column is usable as a category only if it has few unique values,
    # both absolutely and relative to the row count (else it's likely an ID)
    max_cats = min(MAX_CATEGORIES, max(2, len(df) // 2))
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            numeric.append(c)
        elif df[c].nunique() <= max_cats:
            categorical.append(c)
        else:
            dropped.append(c)  # names, ticket numbers, ids...
    if dropped:
        msgs.append("Dropped mostly-unique text column(s) (likely IDs/names, not "
                    f"categories): {', '.join(dropped)}.")
    if not numeric and not categorical:
        return None, None, None, msgs + ["No usable feature columns selected."]

    parts, names = [], []
    if numeric:
        Xn = df[numeric].astype(float)
        n_nan = int(Xn.isna().sum().sum())
        if n_nan:
            Xn = Xn.fillna(Xn.median())
            msgs.append(f"Imputed {n_nan} missing numeric value(s) with column medians.")
        parts.append(Xn)
        names += numeric
    if categorical:
        dummies = pd.get_dummies(df[categorical].astype("string").fillna("missing"),
                                 prefix=categorical)
        parts.append(dummies.astype(float))
        names += list(dummies.columns)
        msgs.append("One-hot encoded categorical column(s): " +
                    ", ".join(f"{c} ({df[c].nunique()} values)" for c in categorical) + ".")
    X = pd.concat(parts, axis=1)[names]

    y = None
    if target_col is not None:
        y_raw = df[target_col]
        mask = y_raw.notna()
        if (~mask).any():
            msgs.append(f"Dropped {int((~mask).sum())} row(s) with a missing target.")
            X, y_raw = X[mask], y_raw[mask]
        if task == "regression":
            if not pd.api.types.is_numeric_dtype(y_raw):
                return None, None, None, msgs + ["Regression needs a numeric target column."]
            y = y_raw.to_numpy(dtype=float)
        else:  # classification: normalize any target to 0..k-1 labels
            y, labels = pd.factorize(y_raw)
            if not pd.api.types.is_numeric_dtype(y_raw):
                msgs.append(f"Encoded target labels as integers: {list(labels)}.")

    X = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return X, y, names, msgs
