from .common_targetters import *
from .common import *
from .continuation import *
from .integrate import *
from .interpolate import *
from .plotters import *
from .utils import *


def _warmup():
    x0 = np.array([0.1, 0.2, 0.0, 0.0])
    tf = 0.1
    Lr, Mr = 1.0, 1.0
    tol = 1e-8
    ts, xs, fs = integrate_state(x0, tf, Lr, Mr, tol)
    _ = integrate_state_stm(x0, tf, Lr, Mr, tol)
    _ = interp_hermite(ts, xs, fs, np.linspace(0.0, tf, 20, True))
    _ = interp_hermite(ts, xs, fs, n_mult=3)
    targ = single_fixed(0, x0[1], 3, 1., 1., 1e-3)
    X = np.array([x0[1],x0[2],x0[3],tf])
    _ = targ.g_dg_stm(X)
    dc_underconstrained(X, targ.g_dg_stm, 1e-2, max_iter=5)
    # any other jitted functions in the call chain


_warmup()
