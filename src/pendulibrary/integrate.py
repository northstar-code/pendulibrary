from numba import njit
from numpy.typing import NDArray
import numpy as np
from numba import float64 as nbfloat64
from numba.typed import List as nbList
from typing import Tuple
import pendulibrary.DOP853_coefs as coefs
from pendulibrary.common import eom, stm_eom


@njit(cache=True)
def integrate_state(
    x0: NDArray,
    tf: float,
    Lr: float,
    Mr: float,
    int_tol: float = 1e-12,
    init_step: float = 1.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    atol = rtol = int_tol
    #  Prepare integrator
    assert tf != 0.0
    forward = tf > 0.0
    t = 0.0
    nx = 4

    K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    # lists to return
    ts = nbList()
    ts.append(0.0)
    xs = nbList()
    xs.append(x0)
    fs = nbList()
    fs.append(eom(0.0, x0, Lr, Mr))
    # initialize
    x = x0.copy()
    h = abs(init_step) if forward else -abs(init_step)

    # Begin loop
    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        # Build integration table
        K[0] = eom(t, x, Lr, Mr)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = eom(t + c * h, x + dy, Lr, Mr)

        # Proposed step
        xnew = x + h * np.dot(K[:-1].T, coefs.B)
        K[-1] = eom(t + h, xnew, Lr, Mr)

        # Error checking
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

        # If we've met error tolerance, take a step
        if error <= 1:
            tnew = t + h

            t = tnew
            x = xnew
            ts.append(t)
            xs.append(x)
            fs.append(K[-1].copy())
        h *= hscale

    # Convert everything to arrays
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
    Lr: float,
    Mr: float,
    int_tol: float = 1e-12,
    init_step: float = 1.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    atol = rtol = int_tol
    #  Prepare integrator
    assert tf != 0.0
    forward = tf > 0.0
    t = 0.0
    nx = 20

    K_ext = np.empty((coefs.N_STAGES_EXTENDED, nx), dtype=np.float64)
    K = K_ext[: coefs.n_stages + 1]

    # lists to return
    ts = nbList()
    ts.append(0.0)
    xs = nbList()
    # Build the STM
    x = np.zeros(nx)
    for i in range(4):
        x[i] = x0[i]
        x[4 + i * 5] = 1.0  # STM components
    xs.append(x)

    # Initialize
    h = abs(init_step) if forward else -abs(init_step)

    # Begin the loop
    while (t < tf) if forward else (t > tf):
        if (t + h > tf) if forward else (t + h < tf):
            h = tf - t

        # Build integration table
        K[0] = stm_eom(t, x, Lr, Mr)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = stm_eom(t + c * h, x + dy, Lr, Mr)

        # Proposed step
        xnew = x + h * np.dot(K[:-1].T, coefs.B)
        K[-1] = stm_eom(t + h, xnew, Lr, Mr)

        # Error checking
        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = 0.0
        for comp in err5:
            err5_norm_2 += comp**2
        err3_norm_2 = 0.0
        for comp in err3:
            err3_norm_2 += comp**2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * nx)

        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        # If we've met error tolerance, take a step
        if error <= 1:
            tnew = t + h

            t = tnew
            x = xnew
            ts.append(t)
            xs.append(x)
        h *= hscale

    # Convert everything to arrays
    nt = len(ts)
    ts_out = np.empty((nt), dtype=np.float64)
    xs_out = np.empty((nx, nt), dtype=np.float64)
    for jj in range(nt):
        xs_out[:, jj] = xs[jj]
        ts_out[jj] = ts[jj]

    return ts_out, xs_out
