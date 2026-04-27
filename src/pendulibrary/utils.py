import numpy as np
from pendulibrary.common import get_A_raw
from scipy.linalg import expm
from pendulibrary.common_targetters import single_fixed
from pendulibrary.targeter import dc_underconstrained

# find somewhere better for these functions probably


def get_x0_linear(
    th1_0: float,
    th2_0: float,
    Lr: float,
    Mr: float,
    step: float = 1e-8,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Get linear IC

    Args:
        th1_0 (float): theta1 at t=0
        th2_0 (float): theta2 at t=0
        Lr (float): Length ratio
        Mr (float): Mass ratio_
        step (float, optional): Step size away from equilibrium. Defaults to 1e-8.

    Returns:
        tuple: state vector (or state vectors, if multiple periodic modes exist), period (or periods)
    """

    # Reference state vector
    Xref = np.array([th1_0, th2_0, 0.0, 0.0])
    A = get_A_raw(Xref, Lr, Mr)  # linearized dynamics near origin

    # both frequencies
    ws = np.linalg.eigvals(A)
    ws = np.unique(np.abs(ws[np.abs(np.real(ws)) < 1e-10]))

    # To check both frequencies, make sure there are 1 or more frequencies and set up a of outputs
    Ts = []
    x0s = []
    assert len(ws) >= 1
    # iterate over each frequency
    for w in ws:
        T = 2 * np.pi / w
        Phi = expm(A * T)
        eigs = np.linalg.eig(Phi)
        vals = eigs.eigenvalues
        # Grab the eigenvectors corresponding to eigenvalues close to 1
        e1, e2 = eigs.eigenvectors[:, np.abs(vals - 1) < 1e-10].T

        # auto-decide which state to force to zero
        force_0 = (
            0 if np.abs(e1[0]) > np.abs(e1[1]) and np.abs(e2[0]) > np.abs(e2[1]) else 1
        )
        # get the delta from reference
        if np.any(np.iscomplex(e1)):
            x0 = np.imag(e1 / e1[force_0])
        else:
            x0 = np.real(e1 / e1[force_0] - e2 / e2[force_0])

        x0 /= np.linalg.norm(x0)
        x0 *= step  # scale the state
        x0 += Xref  # and finally add it back to reference

        Ts.append(T)
        x0s.append(x0.copy())

    if len(x0s) == 1:  # If there's just one, we don't need a list
        return x0, float(T)
    else:
        return np.array(x0s), np.array(Ts)


def get_x0_corrected(
    th1_0: float,
    th2_0: float,
    Lr: float,
    Mr: float,
    step: float = 1e-8,
    int_tol: float = 1e-14,
) -> tuple[np.ndarray, float | np.ndarray, np.ndarray]:
    """Get IC near the equilibrium, but in the full nonlinear system

    Args:
        th1_0 (float): theta1 at t=0
        th2_0 (float): theta2 at t=0
        Lr (float): Length ratio
        Mr (float): Mass ratio_
        step (float, optional): Step size away from equilibrium. Defaults to 1e-8.
        int_tol (float, optional): Integration tolerance. Defaults to 1e-14.

    Returns:
        tuple: state vector (or state vectors, if multiple periodic modes exist), period (or periods), tangent vector (or vectors)
    """

    # Same code as the linear one, except... (see below)
    th0s = [th1_0, th2_0]

    Xref = np.array([th1_0, th2_0, 0.0, 0.0])
    A = get_A_raw(Xref, Lr, Mr)

    # both frequencies
    ws = np.linalg.eigvals(A)
    ws = np.unique(np.abs(ws[np.abs(np.real(ws)) < 1e-10]))

    Ts = []
    x0s = []
    tangents = []
    assert len(ws) >= 1
    for w in ws:
        T = 2 * np.pi / w
        Phi = expm(A * T)
        eigs = np.linalg.eig(Phi)
        vals = eigs.eigenvalues
        e1, e2 = eigs.eigenvectors[:, np.abs(vals - 1) < 1e-10].T

        force_0 = (
            0 if np.abs(e1[0]) > np.abs(e1[1]) and np.abs(e2[0]) > np.abs(e2[1]) else 1
        )
        if np.any(np.iscomplex(e1)):
            x0 = np.imag(e1 / e1[force_0])
        else:
            x0 = np.real(e1 / e1[force_0] - e2 / e2[force_0])

        x0 /= np.linalg.norm(x0)
        x0 *= step
        x0 += Xref

        # run nonlinear correction on obtained IC
        targ = single_fixed(
            ind_fixed=force_0,
            val_fixed=0.0 + th0s[force_0],
            ind_no_enforce=2,
            Lr=Lr,
            Mr=Mr,
            int_tol=int_tol,
        )
        func = targ.g_dg_stm
        X, dG, _ = dc_underconstrained(targ.get_X(x0, T), func, 1e-12, debug=False)
        x0, T = targ.get_x0(X), targ.get_period(X)
        # Done with nonlinear correction
        # Get the tangent vector too
        tangent = np.linalg.svd(dG).Vh[-1]

        Ts.append(T)
        x0s.append(x0.copy())
        tangents.append(tangent.copy())

    if len(x0s) == 1:
        return x0, float(T), tangent
    else:
        return np.array(x0s), np.array(Ts), np.array(tangents)
