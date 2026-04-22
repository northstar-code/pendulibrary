import matplotlib.pyplot as plt
from pendulibrary.interpolate import interp_hermite
import matplotlib.animation as animation
from plotly.express.colors import sample_colorscale
from plotly.graph_objects import Figure, Scatter
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
    try:
        inds_in = np.argwhere(np.any(np.abs(np.array(funcs)) < ylim, axis=0))
        indmax = np.max(inds_in) * 1.05

        ax.set(ylim=(-ylim, ylim), xlim=(0, indmax))
    except ValueError:
        pass
    ax.legend()
    return fig


def compare_fams(
    Xs_new: np.ndarray,
    fam_names: list,
    cmap: str = "hsv",
    figsize: tuple = (6, 6),
    ind_skip: int = 0,
):
    states = [r"$\theta_1$", r"$\theta_2$", r"$\omega_1$", r"$\omega_2$", r"T"]
    states.pop(ind_skip)
    fig, axs = plt.subplots(3, 3, figsize=figsize, sharex="col", sharey="row")

    for jj in range(3):
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
        Xs = np.delete(Xs, ind_skip, 0)
        # Xs = np.delete(Xs, 0, 0)
        for jj in range(3):
            for ii in range(jj + 1):
                axs[jj, ii].plot(Xs[ii + 1], Xs[jj], c=colrs[j])

        axs[0, 0].plot(np.nan, np.nan, c=colrs[j], label=fname)

    for jj in range(3):
        for ii in range(jj + 1):
            axs[jj, ii].plot(Xs_new[:, ii + 1], Xs_new[:, jj], ".k")
            axs[jj, ii].plot(Xs_new[-1, ii + 1], Xs_new[-1, jj], "xk")
            axs[jj, ii].plot(Xs_new[0, ii + 1], Xs_new[0, jj], "ok")
            # if ii != 0:
            #     axs[jj, ii].set(yticklabels=[])
            # if jj != 2:
            #     axs[jj, ii].set(xticklabels=[])

    for jj in range(3):
        axs[jj, 0].set(ylabel=states[jj])
        axs[-1, jj].set(xlabel=states[jj + 1])
    fig.legend()
    # axlg.set_axis_off()
    fig.tight_layout()
    return fig


def compare_fast(
    periods: np.ndarray,
    hamiltonians: np.ndarray,
    filenames: list,
    directory: str = "../database/",
    colormap:str='hsv'
):
    directory = directory.rstrip("/")
    vals_dict = {}
    for fname in filenames:
        data = np.load(f"{directory}/{fname}.npz")
        vals_dict[fname] = dict(T=data["periods"], H=data["hamiltonians"])

    per_range = max(periods) - min(periods)
    ham_range = max(hamiltonians) - min(hamiltonians)
    xlim = (max(periods) + min(periods)) / 2 + np.array([-per_range / 2, per_range / 2])
    ylim = (max(hamiltonians) + min(hamiltonians)) / 2 + np.array(
        [-ham_range / 2, ham_range / 2]
    )

    curve1 = Scatter(
        x=periods,
        y=hamiltonians,
        name="Compare",
        hoverinfo="name",
        mode="lines",
        line=dict(width=1.5, color="white"),
        showlegend=False,
    )

    n = len(vals_dict)
    colors = sample_colorscale(colormap, n) if n > 1 else ["rgb(255,0,0)"]
    curves = [curve1]
    j = 0
    for name, dct in vals_dict.items():
        curve = Scatter(
            x=dct["T"],
            y=dct["H"],
            name=name,
            hoverinfo="name",
            mode="lines",
            line=dict(width=1, color=colors[j]),
        )
        j += 1
        curves.append(curve)

    curveend = Scatter(
        x=periods,
        y=hamiltonians,
        name="Compare",
        hoverinfo="name",
        mode="markers",
        marker=dict(color="white", size=4),
    )

    curves.append(curveend)
    fig = Figure(data=curves)
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        xaxis_range=list(xlim),
        yaxis_range=list(ylim),
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(title="Period"),
        yaxis=dict(title="Hamiltonian"),
        modebar_remove=["autoScale", "lasso2d", "select2d", "toImage"],
        modebar_orientation='v'
    )

    config = dict(displaylogo=False, displayModeBar=True)
    fig.show(config=config)
