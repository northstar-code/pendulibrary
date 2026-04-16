from pendulibrary.integrate import *


def dc_tangent(
    X_prev: NDArray,
    tangent: NDArray,
    f_df_func: Callable,
    s: float = 1e-3,
    tol: float = 1e-8,
    max_iter: int | None = None,
    fudge: float | None = None,
    max_step: float | None = None,
    debug: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, int]:
    """Pseudoarclength continuation differential corrector. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X_prev (NDArray): previous control variables
        tangent (NDArray): tangent to previous orbit. Would be nice to not have to carry over...
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        s (float, optional): step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter (int): maximum number of iterations
        fudge (float): multiply step by this much

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """
    # To be phased out later
    X = X_prev + s * tangent
    if fudge is None:
        fudge = 1.0

    nX = len(X)
    dF = np.empty((nX - 1, nX))
    stm_full = np.empty((nX, nX))

    G = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(G) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        delta = X - X_prev
        lastG = np.dot(delta, delta) - s**2
        lastDG = 2 * delta
        G = np.array([*f, lastG])
        dG = np.vstack((dF, lastDG))
        dX = -np.linalg.inv(dG) @ G
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += dX * fudge
        if debug:
            print(niters, dX)
        niters += 1

    return X, dF, stm_full, niters


def dc_square(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1.0,
    max_step: float | None = None,
    debug: bool = False,
    max_iter: int | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Natural parameter continuation differetial corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter: maximum number of iterations
        fudge (float): multiply step by this much
        debug (bool): whether to print out steps
        max_iter: int|None: how many iterations to cap at


    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """

    X = X_guess.copy()

    nX = len(X)
    dF = np.empty((nX, nX))
    stm_full = np.empty((nX, nX))

    f = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(f) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.inv(dF) @ f
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += fudge * dX
        niters += 1
        if debug:
            print(dX)

    return X, dF, stm_full


def dc_underconstrained(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
    fudge: float = 1.0,
    max_step: float | None = None,
    debug: bool = False,
    max_iter: int | None = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Underconstrained differential corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        max_iter: maximum number of iterations
        fudge (float): multiply step by this much
        debug (bool): whether to print out steps
        max_iter: int|None: how many iterations to cap at

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """
    X = X_guess.copy()
    nX = len(X_guess)
    dF = np.empty((nX, nX))
    stm_full = np.empty((nX, nX))

    f = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(f) > tol and np.linalg.norm(dX) > tol:
        if max_iter is not None and niters > max_iter:
            raise RuntimeError("Exceeded maximum iterations")
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.pinv(dF) @ f
        if max_step is not None and np.linalg.norm(dX) > max_step:
            dX *= max_step / np.linalg.norm(dX)
        X += fudge * dX
        niters += 1
        if debug:
            print(dX)

    return X, dF, stm_full
