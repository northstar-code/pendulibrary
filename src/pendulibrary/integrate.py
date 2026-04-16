from numba import njit
from numpy.typing import NDArray
import numpy as np
from numba import float64 as nbfloat64
from numba.typed import List as nbList
from typing import Tuple, Callable, List
import pendulibrary.DOP853_coefs as coefs
from pendulibrary.interpolate import integrate_interpolate
from pendulibrary.common import eom, stm_eom

# TODO: hardcode the dynamics function in

# @njit
# def integrate_dense_extra(
#     func: Callable,
#     h: float,
#     t0: float,
#     xf: NDArray[np.floating],
#     x0: NDArray[np.floating],
#     f_fin: NDArray[np.floating],
#     K_ext: NDArray,
#     n: int,
#     args:tuple,
# ) -> NDArray[np.floating]:
#     # print(args)
#     """Interpolation for DOP583 requires a couple extra function evaluations. This function does those

#     Args:
#         func (Callable): Function call with signature (t, x, *args)
#         h (float): Step size for the interval
#         t0 (float): Time at the beginning of the interval
#         xf (NDArray[np.floating]): State at the end of the time interval (Nx, )
#         x0 (NDArray[np.floating]): State at the start of the interval (Nx, )
#         f_fin (NDArray[np.floating]): Function evaluation at the end of the time interval
#         K_ext (NDArray): Extended interpolation matrix MUST BE PREFILLED FROM DOP CALL
#         args (tuple): Additional args to func
#         n (int): Number of state vars

#     Returns:
#         NDArray[np.floating]: Filled function evaluation matrix
#     """
#     for smod in range(3):
#         s = smod + coefs.n_stages + 1
#         a = coefs.A_EXTRA[smod]
#         c = coefs.C_EXTRA[smod]
#         dx = np.dot(K_ext[:s].T, a[:s]) * h
#         K_ext[s] = func(t0 + c * h, x0 + dx, *args)

#     F = np.empty((coefs.INTERPOLATOR_POWER, n), dtype=np.float64)

#     f_old = K_ext[0]
#     delta_x = xf - x0

#     F[0] = delta_x
#     F[1] = h * f_old - delta_x
#     F[2] = 2 * delta_x - h * (f_fin + f_old)
#     F[3:] = h * np.dot(coefs.D, K_ext)
#     return F


# @njit
# def integrate(
#     func: Callable[..., NDArray],
#     t_span: Tuple[float, float],
#     x0: NDArray,
#     int_tol: float = 1e-12,
#     atol: float | None = None,
#     rtol: float | None = None,
#     init_step: float = 1.0,
#     events: Callable[..., float] | None = None,
#     n_events: int = 0,
#     directions: List[int] | None = None,
#     terminals: List[int] | None = None,
#     t_eval: NDArray | None = None,
#     dense_output: bool = False,
#     args: Tuple = (),
# ) -> Tuple[
#     NDArray[np.floating],
#     NDArray[np.floating],
#     Tuple[NDArray[np.floating], NDArray[np.floating]],
#     Tuple[List | None, List | None],
# ]:
#     """High order adaptive RK method with interpolation and events location capability.

#     Example calls:
#     ISS orbit:
#     ```
#     ts, xs, _, _ = dop853(gravity_2b, (0., 90.0*3600),
#                           np.array([4193.0, 0, 5290, 0, 7.68452, 0]), 1e-13, args=(3.986e5, ))
#     ```

#     Harmonic Oscillator with returns at maximum and zero and termination at the first ascending zero
#     ```
#     @njit
#     def maxmin(t,x,k): return x[1] # max or min position
#     @njit
#     def zero(t,x,k): return x[0] # oscillator hits zero

#     events_dispatch = dispatch_events(maxmin, zero) # event dispatcher

#     ts, xs, _, (te, xe) = dop853(harmonic_oscillator, (0., 10.), np.array([5., 1.]), 1e-13,
#                                  events=event_dispatch, n_events=2, terminals=[0,1],
#                                  directions=[0,1], t_eval=np.linspace(0, 10, 1000), args=(spring_k, ))
#     ```

#     Args:
#         func (Callable): dynamics function
#         t_span (Tuple[float, float]): beginning and end times MUST BOTH BE FLOATS
#         x0 (NDArray): initial state
#         int_tol (float, optional): Absolute and relative tolerance (will be assigned to both). Defaults to 1e-12
#         atol (float | None, optional): absolute tolerence. Defaults to None. If not None, will override int_tol
#         rtol (float | None, optional): rel tolerence. Defaults to None. If not None, will override int_tol
#         init_step (float, optional): initial step size. Defaults to 1.0.
#         events (Callable | None, optional): Event function dispatcher. This must be a SINGLE function with signature events(ind_event, t, x, *args) -> float. If None, no events.
#         n_events (int, optional): Number of events functions in the dispatcher. If n_events=N, then events will be called as [events(0, t, x, *args), events(1, t, x, *args), ..., events(N, t, x, *args)]. Defaults to 0
#         directions (List | None, optional): ODE event directions. 1 means only trigger when event is increasing, -1 only triggers when decreasing, 0 triggers on both. If the whole list is None, then all events are assigned direction 0
#         terminals (List | None, optional): ODE event terminal counts. Integration will halt at the the termination count specified. 0 means non-terminal event.
#         t_eval (NDArray | None, optional): times to evaluate at. Will interpolate if these are specified. If None, returns only what's evaluated by the RK solver. Defaults to None, meaning all events are non-terminal
#         dense_output (bool, optional): whether to collect interpolators. Doing so slightly increases computation time, so do not do if not necessary. If t_eval is provided or events are non-empty, then this is set to True
#         args (Tuple, optional): additional args to func(t, x, *args). Defaults to ().

#     Returns:
#         NDArray: ts (N, )
#         NDArray: xs (nx, N)
#         Tuple: [NDArray, NDArray]: (EOM function evals (Nx, N), intermediate evaluations (Nt-1, 7, Nx))
#         Tuple: [NDArray, NDArray]: (Event times [Ne x (Nevents, )], Event states [Ne x (nx, Nevents)])
#     """

#     # prepare inputs
#     if atol is None:
#         atol = int_tol
#     if rtol is None:
#         rtol = int_tol

#     if events is not None:
#         assert n_events > 0
#         dense_output = True
#         if directions is None:
#             directions = [0] * n_events
#         else:
#             assert len(directions) == n_events
#         if terminals is None:
#             terminals = [0] * n_events
#         else:
#             assert len(terminals) == n_events
#     else:
#         n_events = 0

#     if t_eval is not None:
#         dense_output = True

#     # %% Prepare integrator
#     halt = False
#     forward = t_span[1] > t_span[0]

#     t0, tf = t_span
#     t = t0
#     nx = x0.shape[0]

#     K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
#     K = K_ext[: coefs.n_stages + 1]

#     ts = nbList()
#     ts.append(t0)

#     xs = nbList()
#     xs.append(x0)

#     fs = nbList()
#     fs.append(func(t0, x0, *args))

#     Fs = nbList()
#     F = np.zeros((coefs.INTERPOLATOR_POWER, nx), np.float64)
#     if dense_output:
#         Fs.append(F)
#         Fs.pop()

#     # %% prepare events

#     if events is not None:
#         t_events = nbList()
#         event_vals = nbList()
#         x_events = nbList()
#         for jj in range(n_events):
#             t_events_i = nbList.empty_list(nbfloat64)
#             t_events.append(t_events_i)
#             event_vals_i = nbList.empty_list(nbfloat64)
#             event_vals_i.append(events(jj, t0, x0, args))
#             event_vals.append(event_vals_i)
#             x_events_i = nbList()
#             x_events_i.append(np.zeros((nx,), dtype=np.float64))
#             x_events_i.pop()
#             x_events.append(x_events_i)
#     else:
#         t_events = None
#         event_vals = None
#         x_events = None

#     # %% initialize
#     x = x0.copy()
#     h = abs(init_step) if forward else -abs(init_step)

#     while (t < tf) if forward else (t > tf):
#         if (t + h > tf) if forward else (t + h < tf):
#             h = tf - t

#         # syntax taken from Scipy implementation
#         # %% take step
#         K[0] = func(t, x, *args)
#         for sm1 in range(coefs.N_STAGES - 1):
#             s = sm1 + 1
#             a = coefs.A[s]
#             c = coefs.C[s]
#             dy = np.dot(K[:s].T, a[:s]) * h
#             K[s] = func(t + c * h, x + dy, *args)

#         xnew = x + h * np.dot(K[:-1].T, coefs.B)

#         K[-1] = func(t + h, xnew, *args)

#         # %% error estimator:
#         scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
#         err5 = np.dot(K.T, coefs.E5) / scale
#         err3 = np.dot(K.T, coefs.E3) / scale
#         err5_norm_2 = np.linalg.norm(err5) ** 2
#         err3_norm_2 = np.linalg.norm(err3) ** 2
#         denom = err5_norm_2 + 0.01 * err3_norm_2
#         error = np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))
#         # END ERROR ESTIMDATOR

#         # %% accept step
#         hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

#         # When the step is accepted
#         if error <= 1:
#             tnew = t + h

#             if dense_output:
#                 # print(args)
#                 F = integrate_dense_extra(func, h, t, xnew, x, K[-1], K_ext, nx, args)
#                 Fs.append(F)

#             # %% Event handling
#             if events is not None:
#                 for jj in range(n_events):
#                     g = events(jj, tnew, xnew, args)
#                     direction = directions[jj]
#                     terminal = terminals[jj]

#                     event_vals[jj].append(g)
#                     ev_vals = event_vals[jj]
#                     # handle event direction here
#                     if (
#                         direction in [0, 1]
#                         and np.sign(ev_vals[-2]) < 0
#                         and np.sign(ev_vals[-1]) >= 0
#                     ) or (
#                         direction in [-1, 0]
#                         and np.sign(ev_vals[-2]) > 0
#                         and np.sign(ev_vals[-1]) <= 0
#                     ):
#                         g0, g1 = ev_vals[-2], ev_vals[-1]
#                         te, xe = interp_event(x,xnew,t,tnew,F,g0,g1,events,jj,atol,1e-1,args)
#                         t_events[jj].append(te)
#                         x_events[jj].append(xe)

#                     if len(t_events[jj]) == terminal and terminal > 0:
#                         halt = True
#                         break

#             # %% step
#             t = tnew
#             x = xnew
#             ts.append(t)
#             xs.append(x)
#             fs.append(K[-1])

#             if halt:
#                 # TODO: if we find a halt, recompute the final timestep
#                 # and then drop events after the halt (if two events were both bracketed)
#                 break

#         h *= hscale

#     # %% end, interpolate
#     if t_eval is not None:
#         if halt:  # Get rid of excess t_eval
#             if forward and t_eval[-1] > t:
#                 t_eval = np.append(t_eval[t_eval <= t], t)
#             elif not forward and t_eval[-1] <= t:
#                 t_eval = np.append(t_eval[t_eval <= t], t)
#         _, xs_out = integrate_interpolate(ts, xs, Fs, t_eval)
#         ts_out = t_eval
#     else:  # convert ts and xs to arrays
#         nt = len(ts)
#         ts_out = np.empty((nt), dtype=np.float64)
#         xs_out = np.empty((nx, nt), dtype=np.float64)
#         for jj in range(nt):
#             xs_out[:, jj] = xs[jj]
#             ts_out[jj] = ts[jj]

#     # convert fs and Fs to arrays
#     fs_out = np.empty((len(ts), nx), dtype=np.float64)
#     Fs_out = (
#         np.empty((len(ts), coefs.INTERPOLATOR_POWER, nx), dtype=np.float64)
#         if dense_output
#         else None
#     )
#     for jj in range(len(ts)):
#         fs_out[jj] = fs[jj]
#     if dense_output:
#         for jj in range(len(ts) - 1):
#             Fs_out[jj] = Fs[jj]

#     if events is not None:
#         te_out = []
#         xe_out = []
#         for jj in range(n_events):
#             ne = len(t_events[jj])
#             te = np.empty((ne,), np.float64)
#             xe = np.empty((ne, nx), np.float64)
#             for kk in range(ne):
#                 te[kk] = t_events[jj][kk]
#                 xe[kk] = x_events[jj][kk]
#             te_out.append(te)
#             xe_out.append(xe)
#     else:
#         te_out = None
#         xe_out = None

#     return (ts_out, xs_out, (fs_out, Fs_out), (te_out, xe_out))


@njit(cache=True)
def integrate_state(
    x0: NDArray,
    tf: float,
    int_tol: float = 1e-12,
    Lr: float = 1.0,
    Mr: float = 1.0,
    init_step: float = 1.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    atol = rtol = int_tol

    # %% Prepare integrator
    assert tf != 0.0
    forward = tf > 0.0

    t = 0.0
    nx = 4

    K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    ts = nbList()
    ts.append(0.)

    xs = nbList()
    xs.append(x0)

    fs = nbList()
    fs.append(eom(0., x0, Lr, Mr))

    # %% initialize
    x = x0.copy()
    h = abs(init_step) if forward else -abs(init_step)

    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        K[0] = eom(t, x, Lr, Mr)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = eom(t + c * h, x + dy, Lr, Mr)

        xnew = x + h * np.dot(K[:-1].T, coefs.B)

        K[-1] = eom(t + h, xnew, Lr, Mr)

        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = 0
        for comp in err5:
            err5_norm_2 += comp**2
        err3_norm_2 = 0
        for comp in err3:
            err3_norm_2 += comp**2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * nx)

        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        if error <= 1:
            tnew = t + h

            t = tnew
            x = xnew
            ts.append(t)
            xs.append(x)
            fs.append(K[-1].copy())
        h *= hscale

    nt = len(ts)
    ts_out = np.empty((nt), dtype=np.float64)
    xs_out = np.empty((nx, nt), dtype=np.float64)
    for jj in range(nt):
        xs_out[:, jj] = xs[jj]
        ts_out[jj] = ts[jj]

    fs_out = np.empty((len(ts), nx), dtype=np.float64)
    for jj in range(len(ts)):
        fs_out[jj] = fs[jj]

    return ts_out, xs_out, fs_out.T


@njit(cache=True)
def integrate_state_stm(
    x0: NDArray,
    tf: float,
    int_tol: float = 1e-12,
    Lr: float = 1.0,
    Mr: float = 1.0,
    init_step: float = 1.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    atol = rtol = int_tol

    # %% Prepare integrator
    assert tf != 0.0
    forward = tf > 0.0

    t = 0.0
    nx = 20

    K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    ts = nbList()
    ts.append(0.0)

    xs = nbList()
    xs.append(x0)

    # %% initialize
    x = np.empty(nx)
    for i in range(4):
        x[i] = x0[i]
        x[4 + i * 5] = 1.0  # STM components
    h = abs(init_step) if forward else -abs(init_step)

    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        K[0] = stm_eom(t, x, Lr, Mr)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = stm_eom(t + c * h, x + dy, Lr, Mr)

        xnew = x + h * np.dot(K[:-1].T, coefs.B)

        K[-1] = stm_eom(t + h, xnew, Lr, Mr)

        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = 0
        for comp in err5:
            err5_norm_2 += comp**2
        err3_norm_2 = 0
        for comp in err3:
            err3_norm_2 += comp**2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * nx)

        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        if error <= 1:
            tnew = t + h

            t = tnew
            x = xnew
            ts.append(t)
            xs.append(x)
        h *= hscale

    nt = len(ts)
    ts_out = np.empty((nt), dtype=np.float64)
    xs_out = np.empty((nx, nt), dtype=np.float64)
    for jj in range(nt):
        xs_out[:, jj] = xs[jj]
        ts_out[jj] = ts[jj]

    return ts_out, xs_out
