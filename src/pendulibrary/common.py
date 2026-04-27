import numpy as np
from numba import njit, prange
from numba import types


# %% DOUBLE PENDULUM


@njit(cache=True)
def eom(_: float, state: np.ndarray, Lr: float, Mr: float) -> np.ndarray:
    """Equation of motion for double pendulum

    Args:
        _ (float): Time (unused for autonomous system)
        state (ndarray): State vector [theta1, theta2, omega1, omega2]
        Lr (float): Length ratio := L2/L1
        Mr (float): Mass ration := m2/m1

    Returns:
        ndarray: Time derivative of state
    """
    th1, th2, dth1, dth2 = state[0], state[1], state[2], state[3]

    # Get common terms
    Dth = th1 - th2
    sinDth = np.sin(Dth)
    cosDth = np.cos(Dth)
    dth1sq = dth1 * dth1
    dth2sq = dth2 * dth2

    # temp storage variables
    den = 1 / (2 + Mr - Mr * np.cos(2 * Dth))
    d1 = (
        -(2 + Mr) * np.sin(th1)
        - Mr * np.sin(th1 - 2 * th2)
        - 2 * Mr * sinDth * (Lr * dth2sq + dth1sq * cosDth)
    ) * den
    d2 = (
        (
            2
            * sinDth
            * (dth1sq * (1 + Mr) + (1 + Mr) * np.cos(th1) + dth2sq * Lr * Mr * cosDth)
        )
        * den
        / Lr
    )

    # output
    dstate = np.empty(4)
    dstate[0] = dth1
    dstate[1] = dth2
    dstate[2] = d1
    dstate[3] = d2
    return dstate


@njit(cache=True)
def get_A_raw(
    state: np.ndarray, Lr: float, Mr: float
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """A(x) matrix overhead function to be called outside of integrator

    Args:
        state (ndarray): State vector [theta1, theta2, omega1, omega2]
        Lr (float): Length ratio := L2/L1
        Mr (float): Mass ration := m2/m1

    Returns:
        np.ndarra: 4x4 Jacobian of dynamics
    """
    Dth = state[0] - state[1]

    sinDth = np.sin(Dth)
    cosDth = np.cos(Dth)
    return get_A(state, sinDth, cosDth, Lr, Mr)


@njit(cache=True)
def get_A(
    state: np.ndarray, sinDth: float, cosDth: float, Lr: float, Mr: float
) -> np.ndarray:
    """Jacobian of dynamics for use in integrators, with some common terms precomputed

    Args:
        state (ndarray): State vector [theta1, theta2, omega1, omega2]
        sinDth (float): sin(theta1-theta2) precomputed in executive dynamcis function
        cosDth (float): cos(theta1-theta2) precomputed in executive dynamcis function
        Lr (float): Length ratio := L2/L1
        Mr (float): Mass ration := m2/m1


    Returns:
        ndarray: 4x4 Jacobian of dynamics
    """
    th1, th2, dth1, dth2 = state

    # first-order common terms for efficiency
    Dth = th1 - th2
    sinDth = np.sin(Dth)
    cosDth = np.cos(Dth)
    sinDthsq = sinDth * sinDth
    sincosDth = sinDth * cosDth
    sin2Dth = 2.0 * sincosDth
    dth1sq = dth1 * dth1
    dth2sq = dth2 * dth2
    Mrp1 = Mr + 1.0
    Mrp2 = Mr + 2.0
    ooL = 1.0 / Lr

    # second-order common terms with some degree of meaning
    th1m2th2 = th1 - 2.0 * th2
    sin_th1m2th2 = np.sin(th1m2th2)
    cos_th1m2th2 = np.cos(th1m2th2)
    sin_th1 = np.sin(th1)
    cos_th1 = np.cos(th1)

    # common terms with no meaning
    den = 2.0 + Mr - Mr * np.cos(2.0 * Dth)
    ooden = 1.0 / den
    oodensq = ooden * ooden

    term0 = Lr * Mr * dth2sq * cosDth + Mrp1 * (dth1sq + cos_th1)
    LrMrdth2sinDth = Lr * Mr * dth2sq * sinDth

    term1 = -2.0 * Mr * (Lr * dth2sq + dth1sq * cosDth)
    term2 = term1 * sinDth - Mr * sin_th1m2th2 - Mrp2 * sin_th1
    term3 = term1 * cosDth - Mr * cos_th1m2th2 - Mrp2 * cos_th1
    term4 = term1 * cosDth - 2.0 * Mr * cos_th1m2th2
    coef = 2.0 * Mr * dth1sq * sinDthsq
    frac1 = -2.0 * Mr * term2 * sin2Dth * oodensq
    MrooLodensq = 4.0 * Mr * ooL * oodensq * term0

    # Allocation
    A = np.zeros((4, 4))

    # Identity block (top-right)
    A[0, 2] = A[1, 3] = 1.0

    # Simple-ish block (bottom-right)
    A[2, 2] = -4.0 * Mr * dth1 * sincosDth * ooden
    A[2, 3] = -4.0 * Lr * Mr * dth2 * sinDth * ooden
    A[3, 2] = 4.0 * dth1 * Mrp1 * sinDth * ooL * ooden
    A[3, 3] = 4.0 * Mr * dth2 * sincosDth * ooden

    # Hessian block (bottom left)
    A[2, 0] = frac1 + (coef + term3) * ooden
    A[2, 1] = -frac1 - (coef + term4) * ooden
    A[3, 0] = (
        2.0
        * ooL
        * ooden
        * ((-LrMrdth2sinDth - Mrp1 * sin_th1) * sinDth + term0 * cosDth)
        - MrooLodensq * sin2Dth * sinDth
    )
    A[3, 1] = (
        2.0 * Mr * dth2sq * sinDthsq * ooden
        - 2.0 * ooL * ooden * term0 * cosDth
        + MrooLodensq * sinDth * sin2Dth
    )

    return A


@njit(cache=True)
def stm_eom(_: float, state: np.ndarray, Lr: float, Mr: float) -> np.ndarray:
    """EOM for state and STM dynamics

    Args:
        _ (float): Time (not used for autonomous dynamics)
        state (ndarray): State vector composed of angle states [theta1, theta2, omega1, omega2] and flattened 4x4 STM
        Lr (float): Length ratio := L2/L1
        Mr (float): Mass ration := m2/m1


    Returns:
        ndarray: 20-vector of derivatives of states and EOMs
    """
    out = np.empty(20)

    # unpack and common terms
    th1, th2, dth1, dth2 = state[:4]
    stm_comps = state[4:]
    Dth = th1 - th2

    sinDth = np.sin(Dth)
    cosDth = np.cos(Dth)
    dth1sq = dth1 * dth1
    dth2sq = dth2 * dth2

    # terms with no meaning
    temp1 = 2 + Mr - Mr * np.cos(2 * Dth)
    den = 1 / temp1

    # state derivatives
    d1 = (
        -(2 + Mr) * np.sin(th1)
        - Mr * np.sin(th1 - 2 * th2)
        - 2 * Mr * sinDth * (Lr * dth2sq + dth1sq * cosDth)
    ) * den
    d2 = (
        (
            2
            * sinDth
            * (dth1sq * (1 + Mr) + (1 + Mr) * np.cos(th1) + dth2sq * Lr * Mr * cosDth)
        )
        * den
        / Lr
    )

    out[0] = dth1
    out[1] = dth2
    out[2] = d1
    out[3] = d2

    # STM derivatives
    A = get_A(state[:4], sinDth, cosDth, Lr, Mr)
    # Manual matrix multipliation (eventually I'll change it to fast-track the identity block and skip the zero block)
    for row in range(4):
        for col in range(4):
            s = 0.0
            for i in range(4):
                s += A[row, i] * stm_comps[col + 4 * i]
            out[4 + row * 4 + col] = s

    return out


def hamiltonian(x: np.ndarray, Lr: float, Mr: float) -> float | np.ndarray:
    """Hamiltonian of one or more state vectors

    Args:
        x (np.ndarray): (4, N) or (4,) vector of states
        Lr (float): Length ratio
        Mr (float): Mass Ratio

    Returns:
        float|np.ndarray: Hamiltonian of eac hstate (if x is 2d) or single state (if x is 1d)
    """
    t1, t2, w1, w2 = x
    dt = t1 - t2
    PE = -(Mr + 1) * np.cos(t1) - Mr * Lr * np.cos(t2)
    KE = (
        (Mr + 1) * w1 * w1 / 2
        + Mr * Lr * Lr * w2 * w2 / 2
        + Mr * Lr * w1 * w2 * np.cos(dt)
    )
    return PE + KE
