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
    if st.checkbox("Ridge", False, key="vr_ridge"):
        members.append(("ridge", _scaled(Ridge(), "ridge")))
    if st.checkbox("Gradient Boosting", False, key="vr_gb"):
        members.append(("gb", GradientBoostingRegressor(random_state=SEED)))
    if not members:  # a voting ensemble needs at least one member
        st.warning("Select at least one member — using Linear Regression.")
        members = [("linear", LinearRegression())]
    return VotingRegressor(members)


def _bagging_reg(n_features=1):
    base_kind = st.selectbox("Base model", ["Decision Tree", "KNN", "Linear", "SVR"],
                             key="bagr_base")
    n = st.slider("n_estimators", 1, 200, 20, key="bagr_n")
    frac = st.slider("max_samples (fraction of rows per model)", 0.1, 1.0, 1.0,
                     key="bagr_s")
    if base_kind == "Decision Tree":
        depth = st.slider("Base tree max_depth", 1, 20, 10, key="bagr_d")
        base = DecisionTreeRegressor(max_depth=depth)
    else:
        base = {"KNN": KNeighborsRegressor(5), "Linear": LinearRegression(),
                "SVR": SVR()}[base_kind]
    return BaggingRegressor(estimator=base, n_estimators=n, max_samples=frac,
                            random_state=SEED)


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
    crit = st.selectbox("Criterion", ["gini", "entropy", "log_loss"], key="dtc_c")
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
    base_kind = st.selectbox("Base model", ["Decision Tree", "KNN", "Logistic", "SVM"],
                             key="bagc_base")
    n = st.slider("n_estimators", 1, 200, 20, key="bagc_n")
    frac = st.slider("max_samples (fraction of rows per model)", 0.1, 1.0, 1.0,
                     key="bagc_s")
    if base_kind == "Decision Tree":
        depth = st.slider("Base tree max_depth", 1, 20, 10, key="bagc_d")
        base = DecisionTreeClassifier(max_depth=depth)
    else:
        base = {"KNN": KNeighborsClassifier(5),
                "Logistic": LogisticRegression(max_iter=2000), "SVM": SVC()}[base_kind]
    return BaggingClassifier(estimator=base, n_estimators=n, max_samples=frac,
                             random_state=SEED)


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
    if st.checkbox("Random Forest", False, key="vc_rf"):
        members.append(("rf", RandomForestClassifier(random_state=SEED)))
    if st.checkbox("SVM (RBF)", False, key="vc_svm"):
        members.append(("svm", _scaled(SVC(probability=True), "svc")))
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

ALGORITHMS["Optimization"] = {
    "Gradient Descent (Linear Regression)": dict(
        task="gd", default="Linear", build=None, extras=set()),
}

if HAS_XGB:
    ALGORITHMS["Classification"]["XGBoost"] = dict(
        task="classification", default="Moons", build=_xgb, extras={"estimators"})


# ============================================================ advanced parameters
# Every remaining sklearn constructor parameter, exposed with a one-line
# explanation (shown as the widget's hover tooltip). Plumbing options
# (n_jobs, verbose, random_state, copy_X, warm_start, cache_size) are fixed
# for reproducibility and left out on purpose.
#
# Widget spec mini-language:
#   ("sel", [options], default)          selectbox (None allowed as an option)
#   ("int", lo, hi, default)             integer slider
#   ("int0", lo, hi, default)            integer slider where 0 means None
#   ("float", lo, hi, default, step)     float slider
#   ("bool", default)                    checkbox
#   ("logf", [choices], default)         select_slider over log-spaced values

TOL = ("logf", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 1e-4)

ADVANCED = {
    "Linear Regression": [
        ("positive", ("bool", False), "Force all coefficients to be >= 0 (e.g. prices, doses)"),
    ],
    "Ridge Regression": [
        ("model__fit_intercept", ("bool", True), "Learn a bias term b; turn off only if data is already centred"),
        ("model__solver", ("sel", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"], "auto"), "Numerical routine used to solve the penalized least squares"),
        ("model__tol", TOL, "Stop iterating once the solution changes less than this"),
        ("model__positive", ("bool", False), "Force all coefficients to be >= 0 (requires lbfgs internally)"),
    ],
    "Lasso Regression": [
        ("model__fit_intercept", ("bool", True), "Learn a bias term b"),
        ("model__selection", ("sel", ["cyclic", "random"], "cyclic"), "Order in which coordinate descent updates coefficients; random can converge faster"),
        ("model__tol", TOL, "Convergence tolerance for coordinate descent"),
        ("model__positive", ("bool", False), "Force all coefficients to be >= 0"),
        ("model__max_iter", ("int", 1000, 100000, 50000), "Cap on coordinate-descent iterations"),
    ],
    "ElasticNet Regression": [
        ("model__fit_intercept", ("bool", True), "Learn a bias term b"),
        ("model__selection", ("sel", ["cyclic", "random"], "cyclic"), "Coefficient update order in coordinate descent"),
        ("model__tol", TOL, "Convergence tolerance"),
        ("model__positive", ("bool", False), "Force all coefficients to be >= 0"),
    ],
    "Polynomial Regression": [
        ("poly__interaction_only", ("bool", False), "Only cross-terms like x1*x2, no pure powers like x1^2"),
        ("model__fit_intercept", ("bool", True), "Learn a bias term b"),
        ("model__positive", ("bool", False), "Force all coefficients to be >= 0"),
    ],
    "SVR": [
        ("svr__degree", ("int", 2, 6, 3), "Degree of the polynomial kernel (ignored by rbf/linear)"),
        ("svr__coef0", ("float", 0.0, 5.0, 0.0, 0.1), "Free term in poly kernel — trades high-order vs low-order influence"),
        ("svr__shrinking", ("bool", True), "Speed heuristic that drops points unlikely to be support vectors"),
        ("svr__tol", TOL, "Optimizer stopping tolerance"),
    ],
    "KNN Regressor": [
        ("knn__p", ("sel", [1, 2], 2), "Minkowski power: 1 = Manhattan distance, 2 = Euclidean"),
        ("knn__algorithm", ("sel", ["auto", "ball_tree", "kd_tree", "brute"], "auto"), "Neighbour-search data structure (speed only, same predictions)"),
        ("knn__leaf_size", ("int", 10, 100, 30), "Tree leaf size — affects search speed/memory, not results"),
    ],
    "Decision Tree Regressor": [
        ("criterion", ("sel", ["squared_error", "friedman_mse", "absolute_error", "poisson"], "squared_error"), "What a 'good split' means: MSE, MAE (robust to outliers), or Poisson deviance for counts"),
        ("splitter", ("sel", ["best", "random"], "best"), "best = strongest split each time; random = a random candidate (adds variance)"),
        ("min_samples_split", ("int", 2, 50, 2), "A node needs at least this many samples before it may split"),
        ("min_weight_fraction_leaf", ("float", 0.0, 0.5, 0.0, 0.01), "Minimum fraction of all samples a leaf must hold"),
        ("max_features", ("sel", [None, "sqrt", "log2"], None), "Features considered per split; fewer = more randomness"),
        ("max_leaf_nodes", ("int0", 0, 200, 0), "Hard cap on leaves, grown best-first (0 = no cap)"),
        ("min_impurity_decrease", ("float", 0.0, 0.5, 0.0, 0.01), "A split must reduce impurity at least this much to happen"),
        ("ccp_alpha", ("float", 0.0, 0.1, 0.0, 0.005), "Cost-complexity pruning strength — bigger prunes more after growing"),
    ],
    "Random Forest Regressor": [
        ("criterion", ("sel", ["squared_error", "absolute_error", "friedman_mse", "poisson"], "squared_error"), "Split-quality measure used inside every tree"),
        ("min_samples_split", ("int", 2, 50, 2), "A node needs at least this many samples before it may split"),
        ("max_features", ("sel", [1.0, "sqrt", "log2"], 1.0), "Features tried per split — the forest's decorrelation knob (1.0 = all)"),
        ("bootstrap", ("bool", True), "Train each tree on a bootstrap sample (off = every tree sees all rows)"),
        ("max_samples", ("float", 0.1, 1.0, 1.0, 0.05), "Fraction of rows drawn for each tree's bootstrap (needs bootstrap on)"),
        ("max_leaf_nodes", ("int0", 0, 200, 0), "Hard cap on leaves per tree (0 = no cap)"),
        ("min_impurity_decrease", ("float", 0.0, 0.5, 0.0, 0.01), "Minimum impurity reduction for a split"),
        ("ccp_alpha", ("float", 0.0, 0.1, 0.0, 0.005), "Prune each tree after growing; bigger = simpler trees"),
    ],
    "Gradient Boosting Regressor": [
        ("loss", ("sel", ["squared_error", "absolute_error", "huber", "quantile"], "squared_error"), "Loss being boosted: MSE, MAE, Huber (outlier-robust mix), or quantile"),
        ("subsample", ("float", 0.3, 1.0, 1.0, 0.05), "Row fraction per tree; < 1.0 = stochastic gradient boosting"),
        ("criterion", ("sel", ["friedman_mse", "squared_error"], "friedman_mse"), "Split-quality measure inside each tree"),
        ("min_samples_split", ("int", 2, 50, 2), "Minimum samples a node needs to split"),
        ("min_samples_leaf", ("int", 1, 30, 1), "Minimum samples each leaf must keep"),
        ("max_features", ("sel", [None, "sqrt", "log2"], None), "Features tried per split"),
        ("alpha", ("float", 0.1, 0.95, 0.9, 0.05), "Quantile for huber/quantile losses (ignored otherwise)"),
        ("n_iter_no_change", ("int0", 0, 50, 0), "Early stopping patience on a validation split (0 = off)"),
        ("validation_fraction", ("float", 0.05, 0.4, 0.1, 0.05), "Data held out for early stopping (only if patience > 0)"),
        ("tol", TOL, "Early-stopping improvement threshold"),
        ("ccp_alpha", ("float", 0.0, 0.1, 0.0, 0.005), "Prune each tree after growing"),
    ],
    "AdaBoost Regressor": [
        ("loss", ("sel", ["linear", "square", "exponential"], "linear"), "How prediction errors are turned into sample re-weights each round"),
    ],
    "Bagging Regressor": [
        ("max_features", ("float", 0.1, 1.0, 1.0, 0.05), "Fraction of FEATURES each base model sees (adds feature randomness)"),
        ("bootstrap", ("bool", True), "Sample rows with replacement (off = without replacement)"),
        ("bootstrap_features", ("bool", False), "Sample features with replacement too"),
    ],
    "Logistic Regression": [
        ("logreg__penalty", ("sel", ["l2", "l1", None], "l2"), "Regularization type: l2 shrinks, l1 zeroes coefficients, None = unregularized"),
        ("logreg__fit_intercept", ("bool", True), "Learn a bias term b"),
        ("logreg__class_weight", ("sel", [None, "balanced"], None), "balanced re-weights classes inversely to their frequency (for imbalance)"),
        ("logreg__tol", TOL, "Optimizer stopping tolerance"),
        ("logreg__max_iter", ("int", 100, 10000, 2000), "Cap on solver iterations"),
    ],
    "KNN Classifier": [
        ("knn__p", ("sel", [1, 2], 2), "Minkowski power: 1 = Manhattan, 2 = Euclidean"),
        ("knn__algorithm", ("sel", ["auto", "ball_tree", "kd_tree", "brute"], "auto"), "Neighbour-search structure (speed only)"),
        ("knn__leaf_size", ("int", 10, 100, 30), "Tree leaf size — speed/memory, not accuracy"),
    ],
    "SVM": [
        ("svc__coef0", ("float", 0.0, 5.0, 0.0, 0.1), "Free term in the poly kernel"),
        ("svc__shrinking", ("bool", True), "Speed heuristic that drops unlikely support vectors"),
        ("svc__class_weight", ("sel", [None, "balanced"], None), "balanced boosts the penalty on rare classes"),
        ("svc__tol", TOL, "Optimizer stopping tolerance"),
    ],
    "Naive Bayes (Gaussian)": [],  # var_smoothing is the whole story (priors take an array)
    "Decision Tree Classifier": [
        ("splitter", ("sel", ["best", "random"], "best"), "best = strongest split; random = a random candidate"),
        ("min_samples_split", ("int", 2, 50, 2), "A node needs at least this many samples to split"),
        ("min_weight_fraction_leaf", ("float", 0.0, 0.5, 0.0, 0.01), "Minimum fraction of all samples per leaf"),
        ("max_features", ("sel", [None, "sqrt", "log2"], None), "Features considered per split"),
        ("max_leaf_nodes", ("int0", 0, 200, 0), "Best-first cap on leaves (0 = no cap)"),
        ("min_impurity_decrease", ("float", 0.0, 0.5, 0.0, 0.01), "Required impurity drop for a split"),
        ("class_weight", ("sel", [None, "balanced"], None), "balanced re-weights classes for imbalance"),
        ("ccp_alpha", ("float", 0.0, 0.1, 0.0, 0.005), "Cost-complexity pruning strength"),
    ],
    "Random Forest Classifier": [
        ("criterion", ("sel", ["gini", "entropy", "log_loss"], "gini"), "Impurity measure used for splits in every tree"),
        ("min_samples_split", ("int", 2, 50, 2), "Minimum samples a node needs to split"),
        ("max_features", ("sel", ["sqrt", "log2", None], "sqrt"), "Features tried per split — the decorrelation knob"),
        ("bootstrap", ("bool", True), "Bootstrap rows per tree (off = all rows for every tree)"),
        ("max_samples", ("float", 0.1, 1.0, 1.0, 0.05), "Row fraction per bootstrap (needs bootstrap on)"),
        ("class_weight", ("sel", [None, "balanced", "balanced_subsample"], None), "Re-weight classes; balanced_subsample recomputes per bootstrap"),
        ("max_leaf_nodes", ("int0", 0, 200, 0), "Leaf cap per tree (0 = no cap)"),
        ("min_impurity_decrease", ("float", 0.0, 0.5, 0.0, 0.01), "Required impurity drop for a split"),
        ("ccp_alpha", ("float", 0.0, 0.1, 0.0, 0.005), "Prune each tree after growing"),
    ],
    "Gradient Boosting Classifier": [
        ("loss", ("sel", ["log_loss", "exponential"], "log_loss"), "log_loss = deviance; exponential makes it behave like AdaBoost"),
        ("subsample", ("float", 0.3, 1.0, 1.0, 0.05), "Row fraction per tree; < 1.0 = stochastic boosting"),
        ("criterion", ("sel", ["friedman_mse", "squared_error"], "friedman_mse"), "Split-quality measure inside each tree"),
        ("min_samples_split", ("int", 2, 50, 2), "Minimum samples a node needs to split"),
        ("min_samples_leaf", ("int", 1, 30, 1), "Minimum samples per leaf"),
        ("max_features", ("sel", [None, "sqrt", "log2"], None), "Features tried per split"),
        ("n_iter_no_change", ("int0", 0, 50, 0), "Early-stopping patience (0 = off)"),
        ("validation_fraction", ("float", 0.05, 0.4, 0.1, 0.05), "Held-out fraction for early stopping"),
        ("tol", TOL, "Early-stopping improvement threshold"),
        ("ccp_alpha", ("float", 0.0, 0.1, 0.0, 0.005), "Prune each tree after growing"),
    ],
    "AdaBoost Classifier": [],  # estimator, n_estimators, learning_rate cover it
    "Bagging Classifier": [
        ("max_features", ("float", 0.1, 1.0, 1.0, 0.05), "Fraction of FEATURES each base model sees"),
        ("bootstrap", ("bool", True), "Sample rows with replacement"),
        ("bootstrap_features", ("bool", False), "Sample features with replacement too"),
    ],
    "Voting Classifier": [],  # members + hard/soft cover it
    "Voting Regressor": [],
    "XGBoost": [
        ("colsample_bytree", ("float", 0.3, 1.0, 1.0, 0.05), "Feature fraction sampled per tree"),
        ("min_child_weight", ("int", 1, 20, 1), "Minimum summed hessian in a leaf — bigger = more conservative splits"),
        ("gamma", ("float", 0.0, 5.0, 0.0, 0.1), "Minimum loss reduction a split must earn (pruning at split time)"),
        ("reg_alpha", ("float", 0.0, 5.0, 0.0, 0.1), "L1 penalty on leaf weights (sparsity)"),
        ("reg_lambda", ("float", 0.0, 10.0, 1.0, 0.5), "L2 penalty on leaf weights (shrinkage)"),
    ],
    "K-Means": [
        ("n_init", ("int", 1, 30, 10), "How many random restarts to try; best inertia wins"),
        ("max_iter", ("int", 50, 1000, 300), "Cap on assign/update iterations per restart"),
        ("tol", TOL, "Stop when centroids move less than this"),
        ("algorithm", ("sel", ["lloyd", "elkan"], "lloyd"), "elkan skips distance computations via the triangle inequality (same result)"),
    ],
    "DBSCAN": [
        ("metric", ("sel", ["euclidean", "manhattan", "chebyshev"], "euclidean"), "Distance used for the eps-neighbourhood"),
        ("algorithm", ("sel", ["auto", "ball_tree", "kd_tree", "brute"], "auto"), "Neighbour-search structure (speed only)"),
        ("leaf_size", ("int", 10, 100, 30), "Tree leaf size — speed/memory, not results"),
    ],
    "Agglomerative Clustering": [
        ("metric", ("sel", ["euclidean", "manhattan", "cosine"], "euclidean"), "Distance between points (ward linkage requires euclidean)"),
    ],
    "PCA": [
        ("whiten", ("bool", False), "Rescale each component to unit variance (useful before some downstream models)"),
        ("svd_solver", ("sel", ["auto", "full", "randomized"], "auto"), "Exact vs randomized decomposition (randomized is faster on wide data)"),
    ],
}


def render_advanced(algo: str, model):
    """Render the 'all sklearn parameters' expander and apply choices via
    set_params. Returns the (possibly reconfigured) model."""
    entries = ADVANCED.get(algo)
    if not entries:
        if entries is not None:  # algorithm known, just nothing left to expose
            st.caption("All of this algorithm's sklearn parameters are already above.")
        return model
    with st.expander("Advanced parameters (all sklearn options)"):
        st.caption("Hover a label's ? for what the parameter does. Plumbing "
                   "options (n_jobs, verbose, random_state) are fixed for "
                   "reproducibility.")
        params = {}
        for name, spec, help_text in entries:
            key = f"adv_{algo}_{name}"
            kind = spec[0]
            label = name.split("__")[-1]
            if kind == "sel":
                params[name] = st.selectbox(label, spec[1],
                                            index=spec[1].index(spec[2]),
                                            key=key, help=help_text,
                                            format_func=lambda v: "None" if v is None else str(v))
            elif kind == "int":
                params[name] = st.slider(label, spec[1], spec[2], spec[3],
                                         key=key, help=help_text)
            elif kind == "int0":
                v = st.slider(f"{label} (0 = None)", spec[1], spec[2], spec[3],
                              key=key, help=help_text)
                params[name] = None if v == 0 else v
            elif kind == "float":
                params[name] = st.slider(label, spec[1], spec[2], spec[3], spec[4],
                                         key=key, help=help_text)
            elif kind == "bool":
                params[name] = st.checkbox(label, spec[1], key=key, help=help_text)
            elif kind == "logf":
                params[name] = st.select_slider(label, spec[1], spec[2], key=key,
                                                help=help_text,
                                                format_func=lambda v: f"{v:g}")
        # compatibility fixups sklearn would otherwise reject
        if algo == "Logistic Regression":
            params["logreg__solver"] = ("saga" if params.get("logreg__penalty") == "l1"
                                        else "lbfgs")
        if algo in ("Random Forest Regressor", "Random Forest Classifier"):
            if not params.get("bootstrap", True) or params.get("max_samples") == 1.0:
                params.pop("max_samples", None)  # only valid with bootstrap
        if algo == "Agglomerative Clustering" and model.linkage == "ward":
            if params.get("metric") != "euclidean":
                st.caption("⚠️ ward linkage requires euclidean — metric ignored.")
            params.pop("metric", None)
        if algo in ("Gradient Boosting Regressor", "Gradient Boosting Classifier"):
            if not params.get("n_iter_no_change"):
                params.pop("validation_fraction", None)  # only used with early stopping
        model.set_params(**params)
    return model
