import matplotlib.pyplot as plt
from pendulibrary.interpolate import interp_hermite
import matplotlib.animation as animation
import numpy as np


def plot_timeline(
    xs_preprop: np.ndarray,
    ts_preprop: np.ndarray,
    fs_preprop: np.ndarray,
    Lr: float,
    N: int,
):
    Tf = ts_preprop[-1]
    _, xs_interp = interp_hermite(
        ts_preprop, xs_preprop, fs_preprop, np.linspace(0, Tf, N, False)
    )

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_preprop[:2]

    fig, axs = plt.subplots(1, N, figsize=(4 * N, 6))

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    for ii in range(N):
        ax = axs[ii]

        ax.plot(
            [0, x1_interp[ii], x2_interp[ii]],
            [0, y1_interp[ii], y2_interp[ii]],
            "o-",
            color="red",
        )
        ax.plot(0, 0, "ok", ms=15)

        ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
        ax.plot(x2_curve, y2_curve, "-k", lw=0.5)
        ax.axis("equal")
        ax.set(xticks=[], yticks=[])
    return fig


def plot_timeline_grid(
    xs_preprop: np.ndarray,
    ts_preprop: np.ndarray,
    fs_preprop: np.ndarray,
    Lr: float,
    nrow: int = 4,
    ncol: int = 7,
):
    Tf = ts_preprop[-1]
    N = nrow * ncol
    _, xs_interp = interp_hermite(
        ts_preprop, xs_preprop, fs_preprop, np.linspace(0, Tf, N, True)
    )
    _, xs_dense = interp_hermite(ts_preprop, xs_preprop, fs_preprop, n_mult=5)

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_dense[:2]

    fig, axs_grid = plt.subplots(nrow, ncol, figsize=(4 * ncol, 6 * nrow))
    axs = axs_grid.ravel()

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    for ii in range(N):
        ax = axs[ii]

        ax.plot(0, 0, "ok", ms=15)

        ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
        ax.plot(x2_curve, y2_curve, "-k", lw=0.5)

        ax.plot(
            [0, x1_interp[ii], x2_interp[ii]],
            [0, y1_interp[ii], y2_interp[ii]],
            "o-",
            color="red",
        )

        ax.axis("equal")
        ax.set(xticks=[], yticks=[])
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def make_gif(
    xs: np.ndarray,
    ts: np.ndarray,
    fs: np.ndarray,
    Lr: float,
    file_name: str,
    frames: int = 100,
    figsize: tuple = (5, 5),
    fps: int = 30,
):
    if not file_name.endswith(".gif"):
        file_name = file_name + ".gif"
    Tf = ts[-1]
    _, xs_interp = interp_hermite(ts, xs, fs, np.linspace(0, Tf, frames, False))
    _, xs_dense = interp_hermite(ts, xs, fs, n_mult=3)
    # xs_dense = xs.copy()

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_dense[:2]

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    fig, ax = plt.subplots(figsize=figsize)
    (line,) = ax.plot(
        [0, x1_interp[0], x2_interp[0]],
        [0, y1_interp[0], y2_interp[0]],
        "o-",
        color="red",
    )
    ax.plot(0, 0, "ok", ms=15)
    ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
    ax.plot(x2_curve, y2_curve, "-k", lw=0.5)
    ax.set(xticks=[], yticks=[])
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.axis("equal")

    def update(frame):
        line.set_xdata([0, x1_interp[frame], x2_interp[frame]])
        line.set_ydata([0, y1_interp[frame], y2_interp[frame]])
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    writer = animation.PillowWriter(fps=fps)
    anim.save(file_name, writer=writer)
    plt.close(fig)
