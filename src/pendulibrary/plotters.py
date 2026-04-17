import matplotlib.pyplot as plt
from pendulibrary.interpolate import interp_hermite
from pendulibrary.integrate import integrate_state
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
