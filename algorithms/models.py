"""Algorithm registry: for each algorithm, which task it solves, its default
dataset, and a `build(n_features)` function that renders its hyperparameter
widgets (in the sidebar) and returns a ready-to-fit estimator."""

import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # keep the app working without xgboost
    HAS_XGB = False

SEED = 0
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _scaled(model, name):
    """Wrap a distance/margin-based model with a StandardScaler."""
    return Pipeline([("scaler", StandardScaler()), (name, model)])


def _poly_degree(label, n_features, default_1d, key):
    # polynomial expansion of many features explodes combinatorially,
    # so cap the degree when the data has more than one feature
    max_d = 12 if n_features == 1 else 3
    return st.slider(label, 1, max_d, min(default_1d, max_d), key=key)


# Each control function renders widgets and returns an unfitted estimator.
# Widget keys are prefixed with the algorithm name so state never collides.

# ---------------------------------------------------------------- regression

def _linear(n_features=1):
    return LinearRegression(fit_intercept=st.checkbox("Fit intercept", True, key="lin_int"))


def _ridge(n_features=1):
    alpha = st.select_slider("alpha (L2 strength)", ALPHAS, 1.0, key="ridge_a")
    degree = _poly_degree("Polynomial degree of features", n_features, 8, "ridge_d")
    return Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                     ("scaler", StandardScaler()),
                     ("model", Ridge(alpha=alpha))])


def _lasso(n_features=1):
    alpha = st.select_slider("alpha (L1 strength)", ALPHAS, 0.01, key="lasso_a")
    degree = _poly_degree("Polynomial degree of features", n_features, 8, "lasso_d")
    return Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                     ("scaler", StandardScaler()),
                     ("model", Lasso(alpha=alpha, max_iter=50_000))])


def _poly(n_features=1):
    degree = _poly_degree("Degree", n_features, 3, "poly_d")
    return Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                     ("scaler", StandardScaler()),
                     ("model", LinearRegression())])


def _depth(label, default, key):
    """max_depth slider with an 'unlimited' option (max_depth=None)."""
    if st.checkbox(f"{label}: no limit", False, key=key + "_none"):
        return None
    return st.slider(label, 1, 30, default, key=key)


def _elastic(n_features=1):
    alpha = st.select_slider("alpha (overall strength)", ALPHAS, 0.01, key="en_a")
    l1 = st.slider("l1_ratio (0 = Ridge ... 1 = Lasso)", 0.0, 1.0, 0.5, 0.05, key="en_l1")
    degree = _poly_degree("Polynomial degree of features", n_features, 8, "en_d")
    return Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                     ("scaler", StandardScaler()),
                     ("model", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=50_000))])


def _svr(n_features=1):
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svr_k")
    C = st.select_slider("C", CS, 1.0, key="svr_c")
    eps = st.slider("epsilon (tube width)", 0.01, 1.0, 0.1, key="svr_e")
    gamma = st.select_slider("gamma", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0, key="svr_g")
    return _scaled(SVR(kernel=kernel, C=C, epsilon=eps, gamma=gamma), "svr")


def _knn_reg(n_features=1):
    k = st.slider("k (neighbours)", 1, 50, 5, key="knnr_k")
    weights = st.selectbox("Weights", ["uniform", "distance"], key="knnr_w")
    return _scaled(KNeighborsRegressor(n_neighbors=k, weights=weights), "knn")


def _tree_reg(n_features=1):
    depth = _depth("max_depth", 4, "dtr_d")
    leaf = st.slider("min_samples_leaf", 1, 30, 1, key="dtr_l")
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                 random_state=SEED)


def _rf_reg(n_features=1):
    n = st.slider("n_estimators", 1, 300, 100, key="rfr_n")
    depth = _depth("max_depth", 10, "rfr_d")
    leaf = st.slider("min_samples_leaf", 1, 30, 1, key="rfr_l")
    return RandomForestRegressor(n_estimators=n, max_depth=depth,
                                 min_samples_leaf=leaf, random_state=SEED)


def _gb_reg(n_features=1):
    n = st.slider("n_estimators", 1, 300, 100, key="gbr_n")
    lr = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.3, 1.0], 0.1, key="gbr_lr")
    depth = st.slider("max_depth", 1, 8, 3, key="gbr_d")
    return GradientBoostingRegressor(n_estimators=n, learning_rate=lr,
                                     max_depth=depth, random_state=SEED)


def _ada_reg(n_features=1):
    n = st.slider("n_estimators", 1, 300, 50, key="adar_n")
    lr = st.select_slider("learning_rate", [0.01, 0.1, 0.5, 1.0, 2.0], 1.0, key="adar_lr")
    depth = st.slider("Base tree max_depth", 1, 10, 3, key="adar_d")
    return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=depth),
                             n_estimators=n, learning_rate=lr, random_state=SEED)


def _voting_reg(n_features=1):
    st.caption("Members (predictions are averaged):")
    members = []
    if st.checkbox("Linear", True, key="vr_lin"):
        members.append(("linear", LinearRegression()))
    if st.checkbox("Decision Tree", True, key="vr_dt"):
        members.append(("tree", DecisionTreeRegressor(max_depth=5, random_state=SEED)))
    if st.checkbox("KNN", True, key="vr_knn"):
        members.append(("knn", _scaled(KNeighborsRegressor(5), "knn")))
    if st.checkbox("SVR (RBF)", False, key="vr_svr"):
        members.append(("svr", _scaled(SVR(), "svr")))
    if not members:  # a voting ensemble needs at least one member
        st.warning("Select at least one member — using Linear Regression.")
        members = [("linear", LinearRegression())]
    return VotingRegressor(members)


def _bagging_reg(n_features=1):
    n = st.slider("n_estimators", 1, 200, 20, key="bagr_n")
    frac = st.slider("max_samples (bootstrap fraction)", 0.1, 1.0, 1.0, key="bagr_s")
    depth = st.slider("Base tree max_depth", 1, 20, 10, key="bagr_d")
    return BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=depth),
                            n_estimators=n, max_samples=frac, random_state=SEED)


# ------------------------------------------------------------ classification

def _logreg(n_features=1):
    C = st.select_slider("C (inverse regularization)", CS, 1.0, key="lr_c")
    return _scaled(LogisticRegression(C=C, max_iter=2000), "logreg")


def _knn_clf(n_features=1):
    k = st.slider("k (neighbours)", 1, 50, 5, key="knnc_k")
    weights = st.selectbox("Weights", ["uniform", "distance"], key="knnc_w")
    return _scaled(KNeighborsClassifier(n_neighbors=k, weights=weights), "knn")


def _svm(n_features=1):
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_k")
    C = st.select_slider("C", CS, 1.0, key="svm_c")
    gamma = st.select_slider("gamma (rbf/poly)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0,
                             key="svm_g")
    degree = st.slider("degree (poly)", 2, 6, 3, key="svm_d")
    return _scaled(SVC(kernel=kernel, C=C, gamma=gamma, degree=degree), "svc")


def _gnb(n_features=1):
    vs = st.select_slider("var_smoothing", [1e-12, 1e-9, 1e-6, 1e-3, 1e-1], 1e-9,
                          key="gnb_vs", format_func=lambda v: f"{v:.0e}")
    return GaussianNB(var_smoothing=vs)


def _tree_clf(n_features=1):
    depth = _depth("max_depth", 3, "dtc_d")
    leaf = st.slider("min_samples_leaf", 1, 30, 1, key="dtc_l")
    crit = st.selectbox("Criterion", ["gini", "entropy"], key="dtc_c")
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                  criterion=crit, random_state=SEED)


def _rf_clf(n_features=1):
    n = st.slider("n_estimators", 1, 300, 100, key="rfc_n")
    depth = _depth("max_depth", 10, "rfc_d")
    leaf = st.slider("min_samples_leaf", 1, 30, 1, key="rfc_l")
    return RandomForestClassifier(n_estimators=n, max_depth=depth,
                                  min_samples_leaf=leaf, random_state=SEED)


def _gb_clf(n_features=1):
    n = st.slider("n_estimators", 1, 300, 100, key="gbc_n")
    lr = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.3, 1.0], 0.1, key="gbc_lr")
    depth = st.slider("max_depth", 1, 8, 3, key="gbc_d")
    return GradientBoostingClassifier(n_estimators=n, learning_rate=lr,
                                      max_depth=depth, random_state=SEED)


def _ada_clf(n_features=1):
    n = st.slider("n_estimators", 1, 300, 50, key="adac_n")
    lr = st.select_slider("learning_rate", [0.01, 0.1, 0.5, 1.0, 2.0], 1.0, key="adac_lr")
    depth = st.slider("Base tree max_depth (1 = stumps)", 1, 10, 1, key="adac_d")
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth),
                              n_estimators=n, learning_rate=lr, random_state=SEED)


def _bagging_clf(n_features=1):
    n = st.slider("n_estimators", 1, 200, 20, key="bagc_n")
    frac = st.slider("max_samples (bootstrap fraction)", 0.1, 1.0, 1.0, key="bagc_s")
    depth = st.slider("Base tree max_depth", 1, 20, 10, key="bagc_d")
    return BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=depth),
                             n_estimators=n, max_samples=frac, random_state=SEED)


def _voting_clf(n_features=1):
    st.caption("Members:")
    members = []
    if st.checkbox("LogReg", True, key="vc_lr"):
        members.append(("logreg", _scaled(LogisticRegression(max_iter=2000), "lr")))
    if st.checkbox("Decision Tree", True, key="vc_dt"):
        members.append(("tree", DecisionTreeClassifier(max_depth=5, random_state=SEED)))
    if st.checkbox("KNN", True, key="vc_knn"):
        members.append(("knn", _scaled(KNeighborsClassifier(5), "knn")))
    if st.checkbox("Naive Bayes", False, key="vc_nb"):
        members.append(("nb", GaussianNB()))
    voting = st.selectbox("Voting", ["hard", "soft"], key="vc_v")
    if not members:
        st.warning("Select at least one member — using Logistic Regression.")
        members = [("logreg", _scaled(LogisticRegression(max_iter=2000), "lr"))]
    return VotingClassifier(members, voting=voting)


def _xgb(n_features=1):
    n = st.slider("n_estimators", 1, 300, 100, key="xgb_n")
    lr = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.3, 1.0], 0.3, key="xgb_lr")
    depth = st.slider("max_depth", 1, 10, 6, key="xgb_d")
    sub = st.slider("subsample", 0.3, 1.0, 1.0, key="xgb_s")
    return XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth,
                         subsample=sub, eval_metric="logloss", random_state=SEED)


# ------------------------------------------------------------------ clustering

def _kmeans(n_features=1):
    k = st.slider("k (clusters)", 1, 10, 3, key="km_k")
    init = st.selectbox("Init", ["k-means++", "random"], key="km_i")
    return KMeans(n_clusters=k, init=init, n_init=10, random_state=SEED)


def _dbscan(n_features=1):
    eps = st.slider("eps (neighbourhood radius)", 0.05, 3.0, 0.3, 0.05, key="db_e")
    ms = st.slider("min_samples", 2, 30, 5, key="db_m")
    return DBSCAN(eps=eps, min_samples=ms)


def _agglo(n_features=1):
    k = st.slider("n_clusters", 2, 10, 3, key="ag_k")
    linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"],
                           key="ag_l")
    return AgglomerativeClustering(n_clusters=k, linkage=linkage)


# ---------------------------------------------------------------- registry

# extras flags drive the extra plots in app.py:
#   coef        coefficient bar chart
#   estimators  train/test score vs n_estimators curve
#   tree        expandable plot_tree diagram (single tree or first ensemble tree)
#   knn_grid    small-multiple boundaries at contrasting k values
#   sv          highlight support vectors
#   elbow       K-Means inertia curve
#   dendrogram  agglomerative dendrogram

ALGORITHMS = {
    "Regression": {
        "Linear Regression": dict(task="regression", default="Linear",
                                  build=_linear, extras={"coef"}),
        "Ridge Regression": dict(task="regression", default="Noisy sine",
                                 build=_ridge, extras={"coef"}),
        "Lasso Regression": dict(task="regression", default="Noisy sine",
                                 build=_lasso, extras={"coef"}),
        "ElasticNet Regression": dict(task="regression", default="Noisy sine",
                                      build=_elastic, extras={"coef"}),
        "Polynomial Regression": dict(task="regression", default="Noisy sine",
                                      build=_poly, extras={"coef"}),
        "SVR": dict(task="regression", default="Noisy sine", build=_svr, extras=set()),
        "KNN Regressor": dict(task="regression", default="Noisy sine",
                              build=_knn_reg, extras=set()),
        "Decision Tree Regressor": dict(task="regression", default="Noisy sine",
                                        build=_tree_reg, extras={"tree"}),
        "Random Forest Regressor": dict(task="regression", default="Noisy sine",
                                        build=_rf_reg, extras={"estimators", "tree"}),
        "Gradient Boosting Regressor": dict(task="regression", default="Noisy sine",
                                            build=_gb_reg, extras={"estimators", "tree"}),
        "AdaBoost Regressor": dict(task="regression", default="Noisy sine",
                                   build=_ada_reg, extras={"estimators", "tree"}),
        "Voting Regressor": dict(task="regression", default="Noisy sine",
                                 build=_voting_reg, extras=set()),
        "Bagging Regressor": dict(task="regression", default="Noisy sine",
                                  build=_bagging_reg, extras={"estimators", "tree"}),
    },
    "Classification": {
        "Logistic Regression": dict(task="classification", default="Blobs",
                                    build=_logreg, extras=set()),
        "KNN Classifier": dict(task="classification", default="Moons",
                               build=_knn_clf, extras={"knn_grid"}),
        "SVM": dict(task="classification", default="Moons", build=_svm,
                    extras={"sv"}),
        "Naive Bayes (Gaussian)": dict(task="classification", default="Blobs",
                                       build=_gnb, extras=set()),
        "Decision Tree Classifier": dict(task="classification", default="Moons",
                                         build=_tree_clf, extras={"tree"}),
        "Random Forest Classifier": dict(task="classification", default="Moons",
                                         build=_rf_clf, extras={"estimators", "tree"}),
        "Gradient Boosting Classifier": dict(task="classification", default="Moons",
                                             build=_gb_clf, extras={"estimators", "tree"}),
        "AdaBoost Classifier": dict(task="classification", default="Moons",
                                    build=_ada_clf, extras={"estimators", "tree"}),
        "Bagging Classifier": dict(task="classification", default="Moons",
                                   build=_bagging_clf, extras={"estimators", "tree"}),
        "Voting Classifier": dict(task="classification", default="Moons",
                                  build=_voting_clf, extras=set()),
    },
    "Clustering": {
        "K-Means": dict(task="clustering", default="Blobs", build=_kmeans,
                        extras={"elbow"}),
        "DBSCAN": dict(task="clustering", default="Circles", build=_dbscan,
                       extras=set()),
        "Agglomerative Clustering": dict(task="clustering", default="Blobs",
                                         build=_agglo, extras={"dendrogram"}),
    },
    "Dimensionality Reduction": {
        "PCA": dict(task="dimred", default="Iris", build=None, extras=set()),
    },
}

if HAS_XGB:
    ALGORITHMS["Classification"]["XGBoost"] = dict(
        task="classification", default="Moons", build=_xgb, extras={"estimators"})
