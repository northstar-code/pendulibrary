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
    nrow: int = 2,
    ncol: int = 10,
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


def plot_timeline_grid(
    xs_preprop: np.ndarray,
    ts_preprop: np.ndarray,
    fs_preprop: np.ndarray,
    Lr: float,
    nrow: int = 2,
    ncol: int = 10,
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


def plot_nu_functions(
    eigs_all: np.ndarray, max_period_mult: int = 5, cmap: str = "hsv"
):
    fig, ax = plt.subplots()
    nus = np.sum(eigs_all, axis=1).real - 2
    colormap = plt.get_cmap(cmap)
    colors = [colormap(j / max_period_mult) for j in range(max_period_mult)]
    funcs = [nus - 2 * np.cos(2 * np.pi / (n + 1)) for n in range(max_period_mult)]
    funcs = np.array(funcs)
    xs = np.arange(len(eigs_all))
    ax.plot([0, xs[-1]], [0, 0], "--")

    for func, c, z in zip(funcs, colors, range(max_period_mult)):
        ax.plot(xs, func, "-", color=c, label=f"P{z+1}")

    ylim = 5
    inds_in = np.argwhere(np.any(np.abs(np.array(funcs)) < ylim, axis=0))
    indmax = np.max(inds_in) * 1.05

    ax.set(ylim=(-ylim, ylim), xlim=(0, indmax))
    ax.legend()
    return fig


def compare_fams(
    Xs_new: np.ndarray, fam_names: list, cmap: str = "hsv", figsize: tuple = (10, 10)
):
    fig, axs = plt.subplots(4, 4)

    for jj in range(4):
        for ii in range(jj):
            axs[-jj - 1, -ii - 1].remove()

    cmp = plt.get_cmap(cmap)
    N = len(fam_names)
    colrs = [cmp(j / N) for j in range(N)]
    for j in range(N):
        fname = fam_names[j]
        data = np.load(f"../database/{fname}.npz")
        x0s = data["x0s"]
        periods = data["periods"]
        Xs = np.column_stack((x0s, periods)).T
        for jj in range(4):
            for ii in range(jj + 1):
                axs[jj, ii].plot(Xs[jj], Xs[ii + 1], c=colrs[j])

    for jj in range(4):
        for ii in range(jj + 1):
            axs[jj, ii].plot(Xs_new[:, jj], Xs_new[:, ii + 1], ".k")
            if ii != 0:
                axs[jj,ii].set(yticklabels=[])
            if jj != 3:
                axs[jj,ii].set(xticklabels=[])
    fig.tight_layout()
    return fig
