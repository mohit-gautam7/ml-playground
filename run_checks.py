"""Verification harness (not part of the app). Drives every algorithm through
the real app with streamlit.testing.AppTest and unit-tests the CSV cleaner."""

import sys

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from algorithms.datasets import prepare_csv
from algorithms.models import ALGORITHMS

failures = []


def fresh(category, algo):
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.radio(key="category").set_value(category)
    at.run()
    at.selectbox(key="algorithm").set_value(algo)
    at.run()
    return at


def report(at, label):
    if at.exception:
        failures.append((label, at.exception[0].value))
        print(f"FAIL  {label}: {at.exception[0].value}")
    else:
        print(f"ok    {label}")


# 1. every algorithm on its default dataset
for category, algos in ALGORITHMS.items():
    for algo in algos:
        report(fresh(category, algo), f"{category} / {algo}")

# 2. non-default datasets: Kaggle/multi-feature/real/shape-sensitive combos
extra_cases = [
    ("Regression", "Linear Regression", "Auto MPG (Kaggle)"),   # multiple linear reg
    ("Regression", "Ridge Regression", "Diabetes"),             # capped poly degree
    ("Regression", "ElasticNet Regression", "Noisy sine"),
    ("Regression", "Gradient Boosting Regressor", "Diamonds (Kaggle)"),
    ("Classification", "Logistic Regression", "Pima Diabetes (Kaggle)"),
    ("Regression", "Random Forest Regressor", "Tips (Kaggle)"),
    ("Regression", "Gradient Boosting Regressor", "Auto MPG (Kaggle)"),
    ("Classification", "Logistic Regression", "Titanic (Kaggle)"),
    ("Classification", "Random Forest Classifier", "Titanic (Kaggle)"),
    ("Classification", "KNN Classifier", "Penguins (Kaggle)"),  # knn_grid on PCA view
    ("Classification", "SVM", "Wine"),                          # sv on PCA view
    ("Classification", "SVM", "Spirals"),
    ("Classification", "Decision Tree Classifier", "Breast Cancer"),
    ("Clustering", "K-Means", "Penguins (Kaggle)"),             # elbow + PCA view
    ("Clustering", "DBSCAN", "Blobs"),
    ("Clustering", "Agglomerative Clustering", "Iris"),
    ("Dimensionality Reduction", "PCA", "Penguins (Kaggle)"),
]
for category, algo, ds in extra_cases:
    at = fresh(category, algo)
    ds_key = [k for k in ("ds_regression", "ds_classification", "ds_clustering",
                          "ds_dimred") if any(s.key == k for s in at.selectbox)][0]
    at.selectbox(key=ds_key).set_value(ds)
    at.run()
    report(at, f"{category} / {algo} on {ds}")

# 2b. new features: feature subsets (incl. 3-D surface path) and unlimited depth
at = fresh("Regression", "Linear Regression")
at.selectbox(key="ds_regression").set_value("Auto MPG (Kaggle)")
at.run()
at.multiselect(key="feat_regression").set_value(["horsepower", "weight"])
at.run()  # exactly 2 features -> 3-D surface path
report(at, "Feature picker / 2-feature 3-D surface (Auto MPG)")

at = fresh("Classification", "Random Forest Classifier")
at.selectbox(key="ds_classification").set_value("Pima Diabetes (Kaggle)")
at.run()
at.multiselect(key="feat_classification").set_value(["Glucose", "BMI", "Age"])
at.run()  # exactly 3 features -> raw-axis 3-D scatter
report(at, "Feature picker / 3-feature 3-D scatter (Pima)")

at = fresh("Classification", "Decision Tree Classifier")
at.checkbox(key="dtc_d_none").set_value(True)
at.run()  # unlimited depth
report(at, "Unlimited max_depth / Decision Tree")

at = fresh("Regression", "Random Forest Regressor")
at.checkbox(key="rfr_d_none").set_value(True)
at.run()
report(at, "Unlimited max_depth / Random Forest Regressor")

# 2c. gradient descent page: default, diverging lr, and low-epoch runs
at = fresh("Optimization", "Gradient Descent (Linear Regression)")
report(at, "Gradient Descent / default")
at.select_slider(key="gd_lr").set_value(1.0)
at.run()
assert not at.exception, at.exception
assert any("learning rate" in w.value.lower() or "diverg" in w.value.lower()
           for w in at.warning), "expected divergence warning at lr=1.0"
print("ok    Gradient Descent / divergence warning at lr=1.0")
at = fresh("Optimization", "Gradient Descent (Linear Regression)")
at.selectbox(key="ds_gd").set_value("Noisy sine")
at.run()
report(at, "Gradient Descent / on Noisy sine")

# 2d. advanced parameters: change a few and confirm the app still runs
at = fresh("Classification", "Decision Tree Classifier")
at.selectbox(key="adv_Decision Tree Classifier_splitter").set_value("random")
at.slider(key="adv_Decision Tree Classifier_min_samples_split").set_value(10)
at.run()
report(at, "Advanced params / DT splitter+min_samples_split")
at = fresh("Classification", "Logistic Regression")
at.selectbox(key="adv_Logistic Regression_logreg__penalty").set_value("l1")
at.run()
report(at, "Advanced params / LogReg l1 penalty (saga fixup)")
at = fresh("Regression", "Random Forest Regressor")
at.checkbox(key="adv_Random Forest Regressor_bootstrap").set_value(False)
at.run()
report(at, "Advanced params / RF bootstrap off (max_samples fixup)")
at = fresh("Clustering", "Agglomerative Clustering")
at.selectbox(key="ag_l").set_value("ward")
at.run()
at.selectbox(key="adv_Agglomerative Clustering_metric").set_value("manhattan")
at.run()
report(at, "Advanced params / ward+manhattan fixup")

# 2e. staged boosting animation expander (GB regressor on 1-D default data)
at = fresh("Regression", "Gradient Boosting Regressor")
assert not at.exception
print("ok    Staged boosting animation renders (GB regressor)")

# 2f. bagging with a non-tree base model
at = fresh("Classification", "Bagging Classifier")
at.selectbox(key="bagc_base").set_value("Logistic")
at.run()
report(at, "Bagging / Logistic base model")

# 3. every CV strategy, on both tasks
cv_cases = [
    ("Classification", "Decision Tree Classifier", "Stratified K-Fold"),
    ("Classification", "Logistic Regression", "K-Fold"),
    ("Classification", "KNN Classifier", "Shuffle Split"),
    ("Classification", "Naive Bayes (Gaussian)", "Leave-One-Out"),
    ("Regression", "Ridge Regression", "K-Fold"),
    ("Regression", "KNN Regressor", "Shuffle Split"),
    ("Regression", "Decision Tree Regressor", "Leave-One-Out"),
]
for category, algo, strategy in cv_cases:
    at = fresh(category, algo)
    at.toggle(key="cv_on").set_value(True)
    at.run()
    at.selectbox(key="cv_strategy").set_value(strategy)
    at.run()
    report(at, f"CV / {algo} / {strategy}")

# LOO guard: 1000 samples must show the too-slow warning, not hang
at = fresh("Classification", "Naive Bayes (Gaussian)")
at.slider(key="n_samples").set_value(1000)
at.run()
at.toggle(key="cv_on").set_value(True)
at.run()
at.selectbox(key="cv_strategy").set_value("Leave-One-Out")
at.run()
assert not at.exception and any("too slow" in w.value for w in at.warning), \
    "expected LOO size warning"
print("ok    CV / LOO guard warns at 1000 samples")

# 4. CSV cleaner: NaNs, categorical encoding, ID-like column, string target
df = pd.DataFrame({
    "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
    "b": [2.0, 1.0, 0.0, 3.0, 5.0, 2.0, 1.0, 0.0],
    "c": list("xyxyxyxy"),                                 # categorical -> one-hot
    "ticket": [f"T{i}{i}{i}" for i in range(8)],           # ID-like -> dropped
    "label": ["cat", "dog", "cat", "dog", "cat", "dog", None, "cat"],
})
X, y, names, msgs = prepare_csv(df, ["a", "b", "c", "ticket"], "label",
                                "classification", )
assert X is not None and X.shape[1] == 4, f"expected a,b + one-hot c_x,c_y: {names}"
assert "c_x" in names and "ticket" not in " ".join(names), names
assert set(np.unique(y)) <= {0, 1} and len(y) == 7
assert any("Imputed" in m for m in msgs) and any("One-hot" in m for m in msgs) \
    and any("mostly-unique" in m for m in msgs), msgs
print("ok    CSV cleaner (impute / one-hot / drop IDs / encode):", msgs)

X, y, names, msgs = prepare_csv(df, ["ticket"], "label", "classification")
assert X is None, "ID-only selection should be rejected with a message"
print("ok    CSV cleaner rejects unusable-only selection")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES"))
sys.exit(1 if failures else 0)
