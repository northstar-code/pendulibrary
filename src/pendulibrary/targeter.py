import numpy as np
from typing import Callable


def dc_tangent(
    X_prev: np.ndarray,
    tangent: np.ndarray,
    f_df_func: Callable,
    s: float = 1e-3,
    tol: float = 1e-8,
    max_iter: int | None = None,
    fudge: float | None = None,
    max_step: float | None = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Tangent-based continuation differential corrector. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X_prev (NDArray): previous control variables
        tangent (NDArray): tangent to previous orbit. Would be nice to not have to carry over...
        f_df_func (Callable): root-finding function with signature f, df/dX, STM = f_df_func(X)
        s (float, optional): step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter (int | None, optional): maximum number of iterations
        fudge (float | None, optional): multiply step by this much
        max_step (float | None, optional): maximum step size within differential correction
        debug (bool, optional): print off debug values

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dG/dx, full-rev STM
    """
    X = X_prev + s * tangent
    if fudge is None:
        fudge = 1.0

    # Initialize variables
    nX = X.shape[0]
    dG = np.empty((nX - 1, nX))
    stm_full = np.empty((nX, nX))
    Gprime = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)

    # begin loop
    while np.linalg.norm(Gprime) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Reached max iterations")
        g, dG, stm_full = f_df_func(X)
        # Get delta between guess and previous known
        delta = X - X_prev
        norm_constraint = np.dot(delta, delta) - s**2
        norm_deriv = 2 * delta
        # augmented cost and its derivative
        Gprime = np.array([*g, norm_constraint])
        dGprime = np.vstack((dG, norm_deriv))
        dX = -np.linalg.inv(dGprime) @ Gprime
        # reduce step size if needed
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += dX * fudge

        if debug:
            print(niters, dX)
        niters += 1

    return X, dG, stm_full, niters


def dc_underconstrained(
    X_guess: np.ndarray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1.0,
    max_step: float | None = None,
    max_iter: int | None = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Underconstrained differential corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): root-finding function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        fudge (float | None, optional): multiply step by this much
        max_step (float | None, optional): maximum step size within differential correction
        max_iter: maximum number of iterations
        debug (bool): whether to print out steps

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dg/dx, trajectory STM
    """
    # initialize variables
    X = X_guess.copy()
    nX = X_guess.shape[0]
    dG = np.empty((nX - 1, nX))
    stm_full = np.empty((nX, nX))
    g = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    # begin loop
    while np.linalg.norm(g) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        g, dG, stm_full = f_df_func(X)
        dX = -np.linalg.pinv(dG) @ g
        # change step size as needed
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)

        X += fudge * dX
        niters += 1
        if debug:
            print(X, dX)

    return X, dG, stm_full
