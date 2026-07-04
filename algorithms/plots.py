"""Matplotlib plotting helpers. Every function returns a Figure; the caller
passes it to st.pyplot and closes it."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def _colors(labels, palette=plt.cm.Set1.colors):
    """Explicit per-point colors so class i always gets palette color i."""
    return np.array(palette)[np.asarray(labels).astype(int) % len(palette)]


def _grid(X, n=250):
    """Mesh grid covering the data with a margin."""
    x0, x1 = X[:, 0].min(), X[:, 0].max()
    y0, y1 = X[:, 1].min(), X[:, 1].max()
    mx, my = 0.1 * (x1 - x0 + 1e-9), 0.1 * (y1 - y0 + 1e-9)
    xx, yy = np.meshgrid(np.linspace(x0 - mx, x1 + mx, n),
                         np.linspace(y0 - my, y1 + my, n))
    return xx, yy


def _boundary_on_ax(ax, model, X_train, y_train, X_test=None, y_test=None):
    xx, yy = _grid(np.vstack([X_train] + ([X_test] if X_test is not None else [])))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int).reshape(xx.shape)
    # one pastel region color per class index, matching the Set1 point colors
    n = int(Z.max()) + 1
    region_cmap = ListedColormap([plt.cm.Pastel1.colors[i % 9] for i in range(n)])
    ax.contourf(xx, yy, Z, alpha=0.7, cmap=region_cmap,
                levels=np.arange(n + 1) - 0.5)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=_colors(y_train),
               s=25, edgecolors="k", linewidths=0.4, label="train")
    if X_test is not None:
        ax.scatter(X_test[:, 0], X_test[:, 1], c=_colors(y_test),
                   s=25, marker="^", edgecolors="k", linewidths=0.4, label="test")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("feature 1"), ax.set_ylabel("feature 2")


def decision_boundary(model, X_train, y_train, X_test, y_test, support_idx=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    _boundary_on_ax(ax, model, X_train, y_train, X_test, y_test)
    if support_idx is not None:  # ring the support vectors
        sv = X_train[support_idx]
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none",
                   edgecolors="k", linewidths=1.2, label="support vectors")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Decision boundary")
    fig.tight_layout()
    return fig


def knn_small_multiples(make_model, X_train, y_train, ks):
    """Boundaries at contrasting k values, fitted fresh for each k."""
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 3.5))
    for ax, k in zip(np.atleast_1d(axes), ks):
        m = make_model(k).fit(X_train, y_train)
        _boundary_on_ax(ax, m, X_train, y_train)
        ax.set_title(f"k = {k}")
    fig.tight_layout()
    return fig


def predicted_vs_actual(model, X_train, y_train, X_test, y_test):
    """For multi-feature regression: predicted-vs-actual plus residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    p_tr, p_te = model.predict(X_train), model.predict(X_test)
    lo = min(y_train.min(), y_test.min(), p_tr.min(), p_te.min())
    hi = max(y_train.max(), y_test.max(), p_tr.max(), p_te.max())
    axes[0].scatter(y_train, p_tr, s=18, alpha=0.6, label="train")
    axes[0].scatter(y_test, p_te, s=18, alpha=0.8, marker="^", label="test")
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    axes[0].set_xlabel("actual y"), axes[0].set_ylabel("predicted y")
    axes[0].set_title("Predicted vs actual"), axes[0].legend(fontsize=8)
    axes[1].scatter(p_te, y_test - p_te, s=18, alpha=0.8, marker="^", color="C1")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("predicted y"), axes[1].set_ylabel("residual (actual − predicted)")
    axes[1].set_title("Test residuals (structure here = something unmodelled)")
    fig.tight_layout()
    return fig


def regression_fit(model, X_train, y_train, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(X_train[:, 0], y_train, s=20, alpha=0.7, label="train")
    ax.scatter(X_test[:, 0], y_test, s=20, alpha=0.7, marker="^", label="test")
    xs = np.linspace(min(X_train.min(), X_test.min()),
                     max(X_train.max(), X_test.max()), 400).reshape(-1, 1)
    ax.plot(xs, model.predict(xs), color="crimson", lw=2, label="model")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.legend(fontsize=8)
    ax.set_title("Fitted curve")
    fig.tight_layout()
    return fig


def coef_bars(names, coefs, title="Coefficients"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(coefs)), coefs, color="steelblue")
    ax.set_xticks(range(len(coefs)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def score_vs_estimators(ns, train_scores, test_scores, score_name):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ns, train_scores, "o-", label="train")
    ax.plot(ns, test_scores, "o-", label="test")
    ax.set_xlabel("n_estimators"), ax.set_ylabel(score_name), ax.legend(fontsize=8)
    ax.set_title(f"{score_name} vs number of estimators")
    fig.tight_layout()
    return fig


def clusters(X, labels, centers=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    noise = labels == -1
    if noise.any():
        ax.scatter(X[noise, 0], X[noise, 1], c="lightgray", marker="x",
                   s=30, label="noise")
        ax.legend(fontsize=8)
    ax.scatter(X[~noise, 0], X[~noise, 1],
               c=_colors(labels[~noise], plt.cm.tab10.colors),
               s=25, edgecolors="k", linewidths=0.3)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200,
                   label="centroids")
        ax.legend(fontsize=8)
    ax.set_xlabel("feature 1"), ax.set_ylabel("feature 2")
    ax.set_title("Cluster assignments")
    fig.tight_layout()
    return fig


def elbow(ks, inertias, current_k):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ks, inertias, "o-")
    ax.axvline(current_k, color="crimson", ls="--", lw=1, label=f"current k={current_k}")
    ax.set_xlabel("k"), ax.set_ylabel("inertia"), ax.legend(fontsize=8)
    ax.set_title("Elbow curve (inertia vs k)")
    fig.tight_layout()
    return fig


def dendrogram_fig(X, linkage_name):
    from scipy.cluster.hierarchy import dendrogram, linkage
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # scipy's ward/complete/average/single names match sklearn's linkage options
    Z = linkage(X, method=linkage_name)
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, no_labels=True)
    ax.set_title(f"Dendrogram ({linkage_name} linkage, last 30 merges)")
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    return fig


def scree(explained_ratio):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    idx = np.arange(1, len(explained_ratio) + 1)
    axes[0].bar(idx, explained_ratio, color="steelblue")
    axes[0].set_xlabel("component"), axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Scree plot")
    axes[1].plot(idx, np.cumsum(explained_ratio), "o-")
    axes[1].axhline(0.95, color="crimson", ls="--", lw=1, label="95%")
    axes[1].set_xlabel("components kept"), axes[1].set_ylabel("cumulative variance")
    axes[1].set_ylim(0, 1.05), axes[1].legend(fontsize=8)
    axes[1].set_title("Cumulative explained variance")
    fig.tight_layout()
    return fig


def pca_projection(X2, y, target_names=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=25,
                    edgecolors="k", linewidths=0.3)
    if target_names is not None:
        handles, _ = sc.legend_elements()
        ax.legend(handles, list(target_names), fontsize=8)
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    ax.set_title("Projection onto first two principal components")
    fig.tight_layout()
    return fig


def tree_diagram(tree, feature_names=None, class_names=None, max_depth=4):
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(tree, ax=ax, filled=True, feature_names=feature_names,
              class_names=class_names, max_depth=max_depth, fontsize=7,
              impurity=False)
    ax.set_title(f"Tree diagram (drawn to depth {max_depth})")
    return fig


# ------------------------------------------------------- interactive 3-D (plotly)

def surface_3d(model, X_train, y_train, X_test, y_test, feat_names):
    """Regression on exactly 2 features: the fitted surface z = f(x1, x2)."""
    import plotly.graph_objects as go
    Xall = np.vstack([X_train, X_test])
    g1 = np.linspace(Xall[:, 0].min(), Xall[:, 0].max(), 60)
    g2 = np.linspace(Xall[:, 1].min(), Xall[:, 1].max(), 60)
    xx, yy = np.meshgrid(g1, g2)
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure([
        go.Surface(x=g1, y=g2, z=zz, opacity=0.6, colorscale="Viridis",
                   showscale=False, name="model"),
        go.Scatter3d(x=X_train[:, 0], y=X_train[:, 1], z=y_train, mode="markers",
                     marker=dict(size=3, color="royalblue"), name="train"),
        go.Scatter3d(x=X_test[:, 0], y=X_test[:, 1], z=y_test, mode="markers",
                     marker=dict(size=4, color="orange", symbol="diamond"),
                     name="test"),
    ])
    fig.update_layout(scene=dict(xaxis_title=feat_names[0], yaxis_title=feat_names[1],
                                 zaxis_title="target y"),
                      height=550, margin=dict(l=0, r=0, t=30, b=0),
                      title="Fitted surface (drag to rotate)")
    return fig


def scatter_3d(X3, labels, axis_names, label_name="class"):
    """Interactive 3-D scatter (e.g. first 3 principal components)."""
    import plotly.express as px
    lab = np.asarray(labels).astype(str)
    lab[lab == "-1"] = "noise"
    fig = px.scatter_3d(x=X3[:, 0], y=X3[:, 1], z=X3[:, 2], color=lab,
                        labels={"x": axis_names[0], "y": axis_names[1],
                                "z": axis_names[2], "color": label_name})
    fig.update_traces(marker=dict(size=3.5))
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=30, b=0),
                      title="3-D view (drag to rotate)")
    return fig
