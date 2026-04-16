import numpy as np
from pendulibrary.common import get_A_raw
from scipy.linalg import expm
from pendulibrary.common_targetters import single_fixed
from pendulibrary.targeter import dc_underconstrained


def get_x0_linear(
    th1_0: float,
    th2_0: float,
    step: float = 1e-8,
    Lr: float = 1.0,
    Mr: float = 1.0,
    int_tol: float = 1e-14,
):
    assert th1_0 == 0 or th2_0 == 0
    force_0 = 0 if th1_0 == 0 else 1

    Xref = np.array([th1_0, th2_0, 0.0, 0.0])
    A = get_A_raw(Xref, Lr, Mr)  # linearized dynamics near origin
    print(A)

    # both frequencies
    ws = np.linalg.eigvals(A)
    ws = np.unique(np.abs(ws[np.abs(np.real(ws)) < 1e-10]))

    Ts = []
    x0s = []
    tangents = []
    assert len(ws) >= 1
    for w in ws:
        T = 2 * np.pi / w
        print(T)
        Phi = expm(A * T)
        eigs = np.linalg.eig(Phi)
        vals = eigs.eigenvalues
        e1, e2 = eigs.eigenvectors[:, np.abs(vals - 1) < 1e-10].T

        x0 = (e1 / e1[force_0] - e2 / e2[force_0]).real
        x0 /= np.linalg.norm(x0)
        print(x0)
        x0 *= step
        x0 += Xref

        targ = single_fixed(
            ind_fixed=force_0,
            val_fixed=0.0,
            ind_no_enforce=0,
            Lr=Lr,
            Mr=Mr,
            int_tol=int_tol,
        )
        func = targ.f_df_stm
        X, dF, _ = dc_underconstrained(targ.get_X(x0, T), func, 1e-12, debug=False)
        x0, T = targ.get_x0(X), targ.get_period(X)
        tangent = np.linalg.svd(dF).Vh[-1]

        Ts.append(T)
        x0s.append(x0.copy())
        tangents.append(tangent.copy())

    if len(x0s) == 1:
        return x0, float(T), tangent
    else:
        return np.array(x0s), np.array(Ts), np.array(tangents)
