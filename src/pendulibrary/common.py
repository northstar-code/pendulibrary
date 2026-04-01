import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from numba import types

from pendulibrary.integrate import dop853
from pendulibrary.interpolate import dop_interpolate


# %% DOUBLE PENDULUM


@njit(cache=True)
def get_A_2pend(
    state: NDArray[np.floating], param: float = 1.0
) -> NDArray[np.floating]:
    Uxx = ...
    O = np.zeros((2, 2))
    I = np.eye(2)
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, O), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def eom_2pend(
    _, state: NDArray[np.floating], param: float = 1.0
) -> NDArray[np.floating]:
    t1, t2, dt1, dt2 = state
    # ..eom here
    ddth = ...

    dstate = np.zeros(4)
    dstate[:2] = state[2:]
    dstate[2:] = ddth
    return dstate


@njit(cache=True)
def stm_eom_2pend(
    _, state: NDArray[np.floating], param: float = 1.0
) -> NDArray[np.floating]:
    pv = state[:4]
    dpv = eom_2pend(0.0, pv, param)
    stm = state[4:].reshape((4, 4))
    A = get_A_2pend(pv, param)
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate


# %% TRIPLE PENDULUM


@njit(cache=True)
def get_A_3pend(
    state: NDArray[np.floating], param: float = 1.0
) -> NDArray[np.floating]:
    Uxx = ...
    O = np.zeros((3, 3))
    I = np.eye(3)
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, O), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def eom_3pend(
    _, state: NDArray[np.floating], param: float = 1.0
) -> NDArray[np.floating]:
    t1, t2, t3, dt1, dt2, dt3 = state
    # ..eom here
    ddth = ...

    dstate = np.zeros(6)
    dstate[:3] = state[3:]
    dstate[3:] = ddth
    return dstate


@njit(cache=True)
def stm_eom_3pend(
    _, state: NDArray[np.floating], param: float = 1.0
) -> NDArray[np.floating]:
    pv = state[:6]
    dpv = eom_3pend(0.0, pv, param)
    stm = state[6:].reshape((6, 6))
    A = get_A_3pend(pv, param)
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate
