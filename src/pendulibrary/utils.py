import numpy as np
from pendulibrary.common import get_A_raw
from scipy.linalg import expm
from pendulibrary.common_targetters import single_fixed
from pendulibrary.targeter import dc_underconstrained


def get_x0_linear(
    th1_0: float,
    th2_0: float,
    Lr: float,
    Mr: float,
    step: float = 1e-8,
    int_tol: float = 1e-14,
):
    th0s = [th1_0, th2_0]

    Xref = np.array([th1_0, th2_0, 0.0, 0.0])
    A = get_A_raw(Xref, Lr, Mr)  # linearized dynamics near origin

    # both frequencies
    ws = np.linalg.eigvals(A)
    ws = np.unique(np.abs(ws[np.abs(np.real(ws)) < 1e-10]))

    Ts = []
    x0s = []
    assert len(ws) >= 1
    for w in ws:
        T = 2 * np.pi / w
        Phi = expm(A * T)
        eigs = np.linalg.eig(Phi)
        vals = eigs.eigenvalues
        e1, e2 = eigs.eigenvectors[:, np.abs(vals - 1) < 1e-10].T

        force_0 = 0 if np.abs(e1[0]) > 0 and np.abs(e2[0]) > 0 else 1
        if np.iscomplex(e1[0]):
            x0 = np.imag(e1 / e1[force_0])
        else:
            x0 = np.real(e1 / e1[force_0] - e2 / e2[force_0])

        x0 /= np.linalg.norm(x0)
        x0 *= step
        x0 += Xref

        Ts.append(T)
        x0s.append(x0.copy())

    if len(x0s) == 1:
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
):
    th0s = [th1_0, th2_0]

    Xref = np.array([th1_0, th2_0, 0.0, 0.0])
    A = get_A_raw(Xref, Lr, Mr)  # linearized dynamics near origin

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

        force_0 = 0 if np.abs(e1[0]) > 0 and np.abs(e2[0]) > 0 else 1
        if np.iscomplex(e1[0]):
            x0 = np.imag(e1 / e1[force_0])
        else:
            x0 = np.real(e1 / e1[force_0] - e2 / e2[force_0])

        x0 /= np.linalg.norm(x0)
        x0 *= step
        x0 += Xref

        print(x0)

        targ = single_fixed(
            ind_fixed=force_0,
            val_fixed=0.0 + th0s[force_0],
            ind_no_enforce=0,
            Lr=Lr,
            Mr=Mr,
            int_tol=int_tol,
        )
        func = targ.f_df_stm
        X, dF, _ = dc_underconstrained(targ.get_X(x0, T), func, 1e-12, debug=False)
        x0, T = targ.get_x0(X), targ.get_period(X)
        print(x0)
        tangent = np.linalg.svd(dF).Vh[-1]

        Ts.append(T)
        x0s.append(x0.copy())
        tangents.append(tangent.copy())

    if len(x0s) == 1:
        return x0, float(T), tangent
    else:
        return np.array(x0s), np.array(Ts), np.array(tangents)
