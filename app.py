"""ML Playground — interactive visual revision tool for classic ML algorithms.

Run with:  streamlit run app.py

All controls live in the sidebar; every change retrains and redraws the main
area immediately. Models always train on the FULL feature set; when data has
more than 2 features, plots use a PCA 2-D view (stated in the UI).
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import (KFold, LeaveOneOut, ShuffleSplit,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)

from algorithms import datasets, gradient_descent as gd, plots
from algorithms.models import ALGORITHMS, HAS_XGB, render_advanced
from algorithms.notes import ANALOGIES, EQUATIONS, LOSSES, NOTES

warnings.filterwarnings("ignore")  # keep sklearn convergence chatter out of the UI

st.set_page_config(page_title="ML Playground", layout="wide")

DATASET_OPTIONS = {
    "regression": ["Noisy sine", "Linear", "Cubic", "Diabetes", "Tips (Kaggle)",
                   "Auto MPG (Kaggle)", "Diamonds (Kaggle)", "Upload CSV"],
    "classification": ["Moons", "Circles", "Blobs", "Spirals",
                       "Two informative features", "Iris", "Wine", "Breast Cancer",
                       "Titanic (Kaggle)", "Penguins (Kaggle)",
                       "Pima Diabetes (Kaggle)", "Upload CSV"],
    "clustering": ["Blobs", "Circles", "Moons", "Anisotropic blobs", "Spirals",
                   "Iris", "Penguins (Kaggle)", "Pima Diabetes (Kaggle)",
                   "Upload CSV"],
    "dimred": ["Iris", "Wine", "Breast Cancer", "Penguins (Kaggle)",
               "Pima Diabetes (Kaggle)", "Upload CSV"],
    "gd": ["Linear", "Noisy sine", "Cubic"],
}
SKLEARN_REAL = {"Iris", "Wine", "Breast Cancer", "Diabetes"}


def show(fig):
    """Render a figure and free its memory."""
    import matplotlib.pyplot as plt
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(show_spinner=False)
def kmeans_inertias(X, k_max=10):
    from sklearn.cluster import KMeans
    ks = list(range(1, k_max + 1))
    return ks, [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_
                for k in ks]


# ============================================================ SIDEBAR: all controls

with st.sidebar:
    st.title("ML Playground")
    category = st.radio("Category", list(ALGORITHMS), key="category")
    algo = st.selectbox("Algorithm", list(ALGORITHMS[category]), key="algorithm")
    spec = ALGORITHMS[category][algo]
    task = spec["task"]
    if not HAS_XGB and category == "Classification":
        st.caption("XGBoost is not installed, so it is hidden from this list.")

    # ------------------------------------------------------- dataset controls
    st.divider()
    st.header("Dataset")
    options = DATASET_OPTIONS[task]
    choice = st.selectbox("Dataset", options, index=options.index(spec["default"]),
                          key=f"ds_{task}")

    X = y = feat_names = None
    msgs = []
    if choice == "Upload CSV":
        file = st.file_uploader("Upload a CSV file", type="csv", key=f"csv_{task}")
        if file is None:
            st.info("Upload a CSV, then pick columns.")
            st.stop()
        df = pd.read_csv(file)
        cols = list(df.columns)
        features = st.multiselect("Feature columns", cols, default=cols[:-1],
                                  key=f"csv_feat_{task}")
        if task in ("regression", "classification"):
            target = st.selectbox("Target column", cols, index=len(cols) - 1,
                                  key=f"csv_tgt_{task}")
        else:
            target = None  # clustering / PCA need no target
        if not features:
            st.warning("Pick at least one feature column.")
            st.stop()
        X, y, feat_names, msgs = datasets.prepare_csv(df, features, target, task)
    elif choice in datasets.KAGGLE_SPECS or choice in SKLEARN_REAL:
        if choice in datasets.KAGGLE_SPECS:
            X, y, feat_names, msgs = datasets.load_kaggle(choice, task)
        else:
            X, y, feat_names, msgs = datasets.load_real(choice)
        # let the student choose which inputs the model actually gets
        picked = st.multiselect("Input features", feat_names, default=feat_names,
                                key=f"feat_{task}")
        if not picked:
            st.warning("Pick at least one input feature.")
            st.stop()
        if len(picked) < len(feat_names):
            keep = [feat_names.index(p) for p in picked]
            X, feat_names = X[:, keep], picked
    else:
        n_samples = st.slider("Samples", 50, 1000, 300, 50, key="n_samples")
        noise = st.slider("Noise level", 0.0, 1.0, 0.2, 0.05, key="noise")
        X, y, feat_names = datasets.make_synthetic(choice, n_samples, noise)

    # -------------------------------------------------- hyperparameter controls
    st.divider()
    st.header("Hyperparameters")
    if task == "dimred":
        model = None
        max_pc = 2 if X is None else max(2, min(X.shape[1], 15))
        n_keep = st.slider("Components to keep", 1, max_pc, 2, key="pca_n")
        pca_model = render_advanced("PCA", PCA(random_state=0))
    elif task == "gd":
        model = None
        gd_lr = st.select_slider("Learning rate η",
                                 [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 0.1,
                                 key="gd_lr")
        gd_epochs = st.slider("Epochs", 5, 300, 60, key="gd_epochs")
        st.caption("Starting point (the parameters GD will change):")
        gd_m0 = st.slider("Initial m (slope)", -100.0, 100.0, -100.0, key="gd_m0")
        gd_b0 = st.slider("Initial b (intercept)", -150.0, 150.0, 120.0, key="gd_b0")
    else:
        model = spec["build"](X.shape[1] if X is not None else 1)
        model = render_advanced(algo, model)

    # -------------------------------------------------------- evaluation controls
    use_cv = False
    if task in ("regression", "classification"):
        st.divider()
        st.header("Evaluation")
        use_cv = st.toggle("Also compute cross-validation", key="cv_on")
        if use_cv:
            strategies = (["Stratified K-Fold", "K-Fold", "Shuffle Split",
                           "Leave-One-Out"] if task == "classification"
                          else ["K-Fold", "Shuffle Split", "Leave-One-Out"])
            cv_strategy = st.selectbox("CV strategy", strategies, key="cv_strategy")
            cv_k = (st.slider("k (folds / splits)", 2, 20, 5, key="cv_k")
                    if cv_strategy != "Leave-One-Out" else None)

# ============================================================ MAIN: everything live

st.title(algo)
if X is None:
    for m in msgs:
        st.error(m)
    st.stop()
for m in msgs:
    st.caption(f"ℹ️ {m}")

intro, points = NOTES[algo]
with st.container(border=True):
    st.markdown(f"*{ANALOGIES[algo]}*")
    st.markdown(f"**How it works.** {intro}")
    st.latex(EQUATIONS[algo])
    st.markdown(f"🎯 **Loss:** {LOSSES[algo]}")
    st.markdown("\n".join(f"- {p}" for p in points))

n_feats = X.shape[1]

# ------------------------------------------------------ gradient descent page

if task == "gd":
    y_gd = y * 40.0  # scale the target so the (m, b) journey is visibly dramatic
    ms, bs, losses = gd.run_gd(X, y_gd, gd_lr, gd_epochs, gd_m0, gd_b0)
    st.subheader("Visualizations")
    st.markdown("**1 — The fit, live.** Press ▶ to watch the line move as the "
                "epochs tick (or scrub the epoch slider):")
    st.plotly_chart(gd.animated_fit(X, y_gd, ms, bs, losses),
                    use_container_width=True)
    st.markdown("**2 — The same journey seen from above.** Every epoch is one "
                "step down the loss contour toward the star (the closed-form "
                "optimum):")
    st.plotly_chart(gd.contour_path(X, y_gd, ms, bs), use_container_width=True)
    st.markdown("**3 — And in 3-D.** The loss surface is a convex bowl; drag to "
                "rotate and watch the path roll to the bottom:")
    st.plotly_chart(gd.surface_path(X, y_gd, ms, bs, losses),
                    use_container_width=True)
    st.markdown("**4 — Training curves.** Loss per epoch, and how m and b "
                "themselves converge:")
    show(gd.history_curves(ms, bs, losses))

    st.subheader("Metrics")
    c = st.columns(3)
    c[0].metric("Final loss (MSE)", f"{losses[-1]:.1f}")
    c[1].metric("Final m, b", f"{ms[-1]:.2f}, {bs[-1]:.2f}")
    m_opt, b_opt = np.polyfit(X[:, 0], y_gd, 1)
    c[2].metric("Closed-form optimum m, b", f"{m_opt:.2f}, {b_opt:.2f}")

    if len(losses) > 2 and losses[-1] > losses[0]:
        st.warning("**What just happened:** the loss went UP — the learning rate "
                   "is too high, so every step overshoots the valley and lands "
                   "higher on the other side. Lower η.")
    elif len(ms) < gd_epochs + 1:
        st.warning("**What just happened:** the parameters exploded to infinity "
                   "(divergence) and training was stopped early. Lower the "
                   "learning rate.")
    elif losses[-1] <= losses[0] and abs(losses[-1] - losses[-2]) / max(losses[0], 1e-9) < 1e-4:
        st.success(f"**What just happened:** converged — the loss curve is flat "
                   f"and (m, b) ≈ ({ms[-1]:.2f}, {bs[-1]:.2f}) matches the "
                   f"closed-form optimum ({m_opt:.2f}, {b_opt:.2f}). More epochs "
                   "would change nothing.")
    else:
        st.info("**What just happened:** the loss is still falling — GD hasn't "
                "converged yet. Add epochs or raise the learning rate a notch.")
    st.stop()

# ---------------------------------------------------------------- PCA page

if task == "dimred":
    if n_feats < 2:
        st.warning("PCA needs at least 2 numeric features.")
        st.stop()
    pca = pca_model.fit(X)  # fit all components once for the scree plot
    ratios = pca.explained_variance_ratio_
    st.subheader("Visualizations")
    show(plots.scree(ratios[:15]))
    yc = y if y is not None else np.zeros(len(X), dtype=int)
    show(plots.pca_projection(pca.transform(X)[:, :2], yc))
    if n_feats >= 3:
        st.plotly_chart(plots.scatter_3d(pca.transform(X)[:, :3], yc,
                                         ["PC1", "PC2", "PC3"]),
                        use_container_width=True)
    st.subheader("Metrics")
    kept = ratios[:n_keep].sum()
    c1, c2 = st.columns(2)
    c1.metric(f"Variance explained by {n_keep} component(s)", f"{kept:.1%}")
    c2.metric("Variance shown in the 2-D plot", f"{ratios[:2].sum():.1%}")
    if kept >= 0.95:
        st.success(f"**What just happened:** {n_keep} components already keep "
                   f"{kept:.1%} of the variance — the rest is nearly redundant.")
    else:
        st.info(f"**What just happened:** {n_keep} components keep {kept:.1%} of the "
                f"variance; the 2-D scatter is a *lossy* view showing "
                f"{ratios[:2].sum():.1%}. If classes overlap there, they may still "
                "be separable in the discarded directions.")
    st.stop()

# --------------------------------------------- 2-D view for plots when needed

viz_pca = None
if task in ("classification", "clustering") and n_feats > 2:
    viz_pca = PCA(n_components=2, random_state=0).fit(X)
    X_viz = viz_pca.transform(X)
elif task in ("classification", "clustering"):
    X_viz = X

# ---------------------------------------------------------------- clustering

if task == "clustering":
    labels = model.fit_predict(X)  # cluster in the FULL feature space
    st.subheader("Visualizations")
    if viz_pca is not None:
        st.caption(f"ℹ️ Clustering ran on all {n_feats} features; the scatter is a "
                   "PCA 2-D view of the result.")
    centers = getattr(model, "cluster_centers_", None)
    if centers is not None and viz_pca is not None:
        centers = viz_pca.transform(centers)
    show(plots.clusters(X_viz, labels, centers))
    if n_feats >= 3:
        if n_feats == 3:
            X3, axes3 = X, list(feat_names)
        else:
            X3 = PCA(n_components=3, random_state=0).fit_transform(X)
            axes3 = ["PC1", "PC2", "PC3"]
        st.plotly_chart(plots.scatter_3d(X3, labels, axes3, "cluster"),
                        use_container_width=True)
    if "elbow" in spec["extras"]:
        show(plots.elbow(*kmeans_inertias(X), model.n_clusters))
    if "dendrogram" in spec["extras"]:
        show(plots.dendrogram_fig(X, model.linkage))

    st.subheader("Metrics")
    clustered = labels != -1  # DBSCAN marks noise as -1
    found = np.unique(labels[clustered])
    c = st.columns(3)
    c[0].metric("Clusters found", len(found))
    if 2 <= len(found) and clustered.sum() > len(found):
        sil = silhouette_score(X[clustered], labels[clustered])
        c[1].metric("Silhouette score", f"{sil:.3f}")
    else:
        sil = None
        c[1].metric("Silhouette score", "n/a")
    noise_frac = 1 - clustered.mean()
    if (labels == -1).any() or algo == "DBSCAN":
        c[2].metric("Noise points", f"{noise_frac:.0%}")

    if len(found) < 2:
        st.warning("**What just happened:** fewer than 2 clusters were found — "
                   "for DBSCAN this usually means eps is too small (all noise) or "
                   "too large (everything merged). Adjust eps/min_samples.")
    elif noise_frac > 0.5:
        st.warning(f"**What just happened:** {noise_frac:.0%} of points were labelled "
                   "noise — eps is likely too small for this data's density.")
    elif sil is not None and sil > 0.5:
        st.success(f"**What just happened:** silhouette {sil:.2f} — points sit much "
                   "closer to their own cluster than to the next one: a clean, "
                   "well-separated clustering.")
    elif sil is not None:
        st.info(f"**What just happened:** silhouette {sil:.2f} — clusters overlap or "
                "their shape doesn't match this algorithm's assumptions. Try another "
                "k / eps / linkage, or a different algorithm for this shape.")
    st.stop()

# ------------------------------------------------------ supervised pipeline

idx = np.arange(len(X))
strat = y if task == "classification" else None
try:
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=1,
                                      stratify=strat)
except ValueError:  # a class with a single sample cannot be stratified
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=1)
X_train, X_test, y_train, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

model.fit(X_train, y_train)  # the metrics model always uses ALL features

st.subheader("Visualizations")

if task == "classification":
    # the boundary plot needs 2-D inputs; with >2 features, refit a copy of the
    # model on the 2 principal components FOR DISPLAY ONLY
    if viz_pca is not None:
        st.caption(f"ℹ️ Metrics use all {n_feats} features. The boundary below is a "
                   "copy of this model refit on 2 principal components, so you can "
                   "still see its shape.")
        viz_model = clone(model).fit(X_viz[tr_idx], y_train)
    else:
        viz_model = model
    support = None
    if "sv" in spec["extras"] and hasattr(viz_model, "named_steps"):
        support = viz_model.named_steps["svc"].support_
    show(plots.decision_boundary(viz_model, X_viz[tr_idx], y_train,
                                 X_viz[te_idx], y_test, support))
    if n_feats >= 3:
        if n_feats == 3:
            X3, axes3 = X, list(feat_names)
        else:
            X3 = PCA(n_components=3, random_state=0).fit_transform(X)
            axes3 = ["PC1", "PC2", "PC3"]
        st.plotly_chart(plots.scatter_3d(X3, y, axes3), use_container_width=True)
    if "knn_grid" in spec["extras"]:
        ks = sorted({1, 15, min(45, len(tr_idx) - 1)})
        st.markdown("**Same data, contrasting k values** — watch the boundary go "
                    "from memorization to over-smoothing:")
        show(plots.knn_small_multiples(
            lambda k: clone(viz_model).set_params(knn__n_neighbors=k),
            X_viz[tr_idx], y_train, ks))
else:
    if n_feats == 1:
        show(plots.regression_fit(model, X_train, y_train, X_test, y_test))
    elif n_feats == 2:
        st.caption("ℹ️ Two input features — the whole model is visible as a 3-D "
                   "surface. Drag to rotate.")
        st.plotly_chart(plots.surface_3d(model, X_train, y_train, X_test, y_test,
                                         feat_names), use_container_width=True)
        show(plots.predicted_vs_actual(model, X_train, y_train, X_test, y_test))
    else:
        st.caption(f"ℹ️ Multiple regression on {n_feats} features — a single fitted "
                   "curve can't be drawn, so here are predicted-vs-actual and "
                   "residual plots instead. Tip: pick exactly 2 input features in "
                   "the sidebar to see the model as a rotatable 3-D surface.")
        show(plots.predicted_vs_actual(model, X_train, y_train, X_test, y_test))

if "coef" in spec["extras"]:
    if hasattr(model, "named_steps"):  # Ridge / Lasso / Polynomial pipelines
        names = list(model.named_steps["poly"].get_feature_names_out(feat_names))
        coefs = model.named_steps["model"].coef_
        title = "Coefficients (on standardized polynomial features)"
    else:  # plain (multiple) linear regression
        names, coefs = list(feat_names), model.coef_
        title = "Coefficients (features are standardized)" if n_feats > 1 \
            else "Coefficient"
    if len(coefs) > 20:  # keep the chart readable on wide feature sets
        top = np.argsort(np.abs(coefs))[::-1][:20]
        names, coefs = [names[i] for i in top], coefs[top]
        title += " — top 20 by |value|"
    show(plots.coef_bars(names, coefs, title))
    if np.any(coefs == 0):
        st.caption(f"{int(np.sum(coefs == 0))} shown coefficient(s) are exactly zero.")

if "estimators" in spec["extras"]:
    n_max = model.get_params()["n_estimators"]
    ns = sorted(set(np.linspace(1, n_max, 6).astype(int)))
    scorer = accuracy_score if task == "classification" else r2_score
    tr, te = [], []
    for n in ns:  # refit at checkpoints to trace the learning behaviour
        m = clone(model).set_params(n_estimators=n).fit(X_train, y_train)
        tr.append(scorer(y_train, m.predict(X_train)))
        te.append(scorer(y_test, m.predict(X_test)))
    show(plots.score_vs_estimators(
        ns, tr, te, "accuracy" if task == "classification" else "R²"))

if hasattr(model, "staged_predict") and n_feats == 1 and task == "regression":
    with st.expander("▶ Animate boosting stages (watch the fit build tree by tree)"):
        st.plotly_chart(plots.staged_fit_animation(model, X_train, y_train),
                        use_container_width=True)

if "tree" in spec["extras"]:
    with st.expander("Show tree diagram" +
                     ("" if "estimators" not in spec["extras"]
                      else " (first tree of the ensemble)")):
        from sklearn.tree import BaseDecisionTree
        tree = (model if isinstance(model, BaseDecisionTree)
                else np.asarray(model.estimators_).ravel()[0])
        if isinstance(tree, BaseDecisionTree):
            class_names = ([str(c) for c in np.unique(y)]
                           if task == "classification" else None)
            show(plots.tree_diagram(tree, list(feat_names), class_names))
        else:  # bagging with a non-tree base model has no tree to draw
            st.caption(f"The base model here is {type(tree).__name__}, "
                       "not a decision tree — nothing to draw.")

# ------------------------------------------------------------- metrics

st.subheader("Metrics")
col_holdout, col_cv = st.columns(2)

with col_holdout:
    st.markdown("**Hold-out split (75% train / 25% test)**")
    yhat_tr, yhat_te = model.predict(X_train), model.predict(X_test)
    if task == "classification":
        acc_tr, acc_te = accuracy_score(y_train, yhat_tr), accuracy_score(y_test, yhat_te)
        c = st.columns(2)
        c[0].metric("Train accuracy", f"{acc_tr:.3f}")
        c[1].metric("Test accuracy", f"{acc_te:.3f}")
        c = st.columns(2)
        c[0].metric("Train F1 (weighted)",
                    f"{f1_score(y_train, yhat_tr, average='weighted'):.3f}")
        c[1].metric("Test F1 (weighted)",
                    f"{f1_score(y_test, yhat_te, average='weighted'):.3f}")
        primary_tr, primary_te = acc_tr, acc_te
    else:
        r2_tr, r2_te = r2_score(y_train, yhat_tr), r2_score(y_test, yhat_te)
        c = st.columns(2)
        c[0].metric("Train R²", f"{r2_tr:.3f}")
        c[1].metric("Test R²", f"{r2_te:.3f}")
        c = st.columns(2)
        c[0].metric("Test MAE", f"{mean_absolute_error(y_test, yhat_te):.3f}")
        c[1].metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, yhat_te)):.3f}")
        primary_tr, primary_te = r2_tr, r2_te

with col_cv:
    if use_cv:
        scoring = "accuracy" if task == "classification" else "r2"
        st.markdown(f"**{cv_strategy} cross-validation ({scoring})**")
        try:
            if cv_strategy == "Leave-One-Out":
                if len(X) > 300:
                    st.warning("Leave-One-Out fits one model per sample — too slow "
                               f"for {len(X)} rows here. Reduce the sample slider "
                               "to ≤ 300 or pick another strategy.")
                else:
                    # single-sample folds have no per-fold R²/accuracy, so pool
                    # the held-out predictions and score them once
                    preds = cross_val_predict(clone(model), X, y, cv=LeaveOneOut())
                    score = (accuracy_score(y, preds) if task == "classification"
                             else r2_score(y, preds))
                    st.metric(f"LOO score (pooled over {len(X)} fits)", f"{score:.3f}")
                    st.caption("Every point is predicted by a model trained on all "
                               "the *other* points — nearly unbiased, but expensive "
                               "and high-variance.")
            else:
                cv = {"K-Fold": KFold(cv_k, shuffle=True, random_state=0),
                      "Stratified K-Fold": StratifiedKFold(cv_k, shuffle=True,
                                                           random_state=0),
                      "Shuffle Split": ShuffleSplit(cv_k, test_size=0.25,
                                                    random_state=0)}[cv_strategy]
                scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring)
                st.metric("CV score", f"{scores.mean():.3f} ± {scores.std():.3f}")
                st.caption("Folds: " + ", ".join(f"{s:.3f}" for s in scores))
                st.caption("The hold-out test score is a *single* random split; CV "
                           "averages several, so it is a steadier estimate of "
                           "generalization.")
        except ValueError as e:  # e.g. stratified folds > smallest class size
            st.warning(f"This CV setting doesn't fit the data: {e}")
    else:
        st.caption("Turn on cross-validation in the sidebar to compare a single "
                   "split against CV.")

# --------------------------------------------------- what just happened

name = "accuracy" if task == "classification" else "R²"
gap = primary_tr - primary_te
low = 0.75 if task == "classification" else 0.5
if primary_tr > 0.9 and gap > 0.12:
    st.warning(f"**What just happened:** train {name} ({primary_tr:.2f}) is far above "
               f"test {name} ({primary_te:.2f}) — the model is **overfitting**: it "
               "memorized the training data. Reduce complexity (lower depth/degree/k⁻¹, "
               "raise regularization) or add data.")
elif primary_tr < low and primary_te < low:
    st.warning(f"**What just happened:** both train ({primary_tr:.2f}) and test "
               f"({primary_te:.2f}) {name} are low — the model is **underfitting**: "
               "it is too constrained for this data. Increase complexity or reduce "
               "regularization.")
else:
    st.success(f"**What just happened:** train {name} {primary_tr:.2f} vs test "
               f"{primary_te:.2f} — a reasonable bias–variance balance; the model "
               "generalizes about as well as it fits.")
