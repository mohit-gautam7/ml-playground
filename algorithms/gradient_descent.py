"""Gradient Descent playground: batch GD fitting y = m*x + b by minimizing MSE,
with the full training history visualized (animated fit, contour path, loss
curves, and a 3-D loss surface with the descent path)."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(show_spinner=False)
def run_gd(X, y, lr, epochs, m0, b0):
    """Full-batch gradient descent on MSE. Returns per-epoch (m, b, loss)
    including the initial state at index 0."""
    x = X[:, 0]
    n = len(x)
    m, b = float(m0), float(b0)
    ms, bs, losses = [m], [b], [float(np.mean((y - (m * x + b)) ** 2))]
    for _ in range(epochs):
        err = y - (m * x + b)
        m -= lr * (-2.0 / n) * np.sum(x * err)   # dL/dm
        b -= lr * (-2.0 / n) * np.sum(err)       # dL/db
        if not np.isfinite(m) or abs(m) > 1e8:   # freeze cleanly on divergence
            break
        ms.append(m), bs.append(b)
        losses.append(float(np.mean((y - (m * x + b)) ** 2)))
    return np.array(ms), np.array(bs), np.array(losses)


def _frame_ids(n, max_frames=60):
    """Subsample epochs so animations stay smooth."""
    return np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))


def animated_fit(X, y, ms, bs, losses):
    """Play-button animation: the fitted line moving epoch by epoch."""
    x = X[:, 0]
    xs = np.array([x.min(), x.max()])
    ids = _frame_ids(len(ms))
    frames = [go.Frame(
        data=[go.Scatter(x=xs, y=ms[i] * xs + bs[i], mode="lines",
                         line=dict(color="crimson", width=3))],
        name=str(i),
        layout=go.Layout(title=f"epoch {i}   m={ms[i]:.2f}  b={bs[i]:.2f}  "
                               f"loss={losses[i]:.1f}"),
        traces=[1]) for i in ids]
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, mode="markers",
                         marker=dict(size=5, color="royalblue"), name="data"),
              go.Scatter(x=xs, y=ms[0] * xs + bs[0], mode="lines",
                         line=dict(color="crimson", width=3), name="fit")],
        frames=frames)
    fig.update_layout(
        height=430, margin=dict(l=0, r=0, t=40, b=0),
        title=f"epoch 0   m={ms[0]:.2f}  b={bs[0]:.2f}  loss={losses[0]:.1f}",
        updatemenus=[dict(type="buttons", showactive=False, y=1.15, x=0,
                          xanchor="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=80,
                                                                    redraw=False),
                                                         fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(mode="immediate")])])],
        sliders=[dict(steps=[dict(method="animate", label=str(i),
                                  args=[[str(i)], dict(mode="immediate",
                                                       frame=dict(duration=0,
                                                                  redraw=False))])
                             for i in ids],
                      currentvalue=dict(prefix="epoch "))])
    return fig


def _loss_grid(X, y, ms, bs, span=1.4, n=80):
    """MSE over a (m, b) grid covering the descent path and the optimum."""
    x = X[:, 0]
    m_opt = np.polyfit(x, y, 1)[0]
    b_opt = np.polyfit(x, y, 1)[1]
    m_lo, m_hi = min(ms.min(), m_opt), max(ms.max(), m_opt)
    b_lo, b_hi = min(bs.min(), b_opt), max(bs.max(), b_opt)
    m_pad, b_pad = span * (m_hi - m_lo + 1), span * (b_hi - b_lo + 1)
    mg = np.linspace(m_lo - m_pad, m_hi + m_pad, n)
    bg = np.linspace(b_lo - b_pad, b_hi + b_pad, n)
    MM, BB = np.meshgrid(mg, bg)
    # loss(m, b) = mean((y - m x - b)^2), vectorized over the grid
    L = np.mean((y[None, None, :] - MM[..., None] * x[None, None, :]
                 - BB[..., None]) ** 2, axis=-1)
    return mg, bg, L, (m_opt, b_opt)


def contour_path(X, y, ms, bs):
    """Loss contour in (m, b) space with the descent path drawn on top."""
    mg, bg, L, (m_opt, b_opt) = _loss_grid(X, y, ms, bs)
    fig = go.Figure([
        go.Contour(x=mg, y=bg, z=L, colorscale="Viridis", showscale=False,
                   contours=dict(coloring="fill")),
        go.Scatter(x=ms, y=bs, mode="lines+markers",
                   line=dict(color="white", width=2), marker=dict(size=4),
                   name="descent path"),
        go.Scatter(x=[ms[0]], y=[bs[0]], mode="markers",
                   marker=dict(size=10, color="red"), name="start"),
        go.Scatter(x=[m_opt], y=[b_opt], mode="markers",
                   marker=dict(size=10, color="gold", symbol="star"),
                   name="optimum"),
    ])
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=40, b=0),
                      title="Loss contour over (m, b) — the bowl seen from above",
                      xaxis_title="m (slope)", yaxis_title="b (intercept)")
    return fig


def surface_path(X, y, ms, bs, losses):
    """3-D loss surface z = MSE(m, b) with the descent path on it."""
    mg, bg, L, _ = _loss_grid(X, y, ms, bs)
    ids = _frame_ids(len(ms), 80)
    fig = go.Figure([
        go.Surface(x=mg, y=bg, z=L, colorscale="Plasma", opacity=0.85,
                   showscale=False),
        go.Scatter3d(x=ms[ids], y=bs[ids], z=losses[ids] * 1.02,
                     mode="lines+markers", marker=dict(size=4, color="royalblue"),
                     line=dict(color="white", width=3), name="descent path"),
    ])
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0),
                      title="Loss surface (drag to rotate) — the ball rolls downhill",
                      scene=dict(xaxis_title="m (slope)", yaxis_title="b (intercept)",
                                 zaxis_title="MSE loss"))
    return fig


def history_curves(ms, bs, losses):
    """Loss vs epoch and parameter values vs epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ep = np.arange(len(losses))
    axes[0].plot(ep, losses, color="C0")
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Loss vs epoch")
    axes[1].plot(ep, ms, label="m (slope)")
    axes[1].plot(ep, bs, label="b (intercept)")
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("parameter value")
    axes[1].set_title("Parameters vs epoch"), axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig
