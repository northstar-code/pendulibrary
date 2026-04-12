import numpy as np
from pendulibrary.integrate import integrate
from numpy.typing import NDArray
from typing import Tuple, List
from abc import ABC, abstractmethod

"""
Targetter objects are used in family continuation, differential correction, and conversion between dynamical systems

Each targetter has methods to get control variables (X) given initial state and period, and another pair of methods to go backward

Targetters all have some way of getting the cost function and its Jacobian

The aforementioned methods are all combined in f_df_stm, which takes a control variable (and possibly more args and kwargs) and
return the cost function, its Jacobian, and STM. In family continuation, this is the only function that should be the only integration instance called.
Some targetters (i.e. heterclinic targetter) are less standard, but still do have these methods

Returns:
    _type_: _description_
"""

# TODO: update these to be specific to our use case


class Targetter(ABC):
    @abstractmethod
    def __init__(self, int_tol: float, param: float, *args, **kwargs):
        self.mu = np.nan
        self.int_tol = np.nan

    @abstractmethod
    def get_X(self, x0: NDArray, period: float) -> NDArray:
        pass

    @abstractmethod
    def get_x0(self, X: NDArray) -> NDArray:
        pass

    @abstractmethod
    def get_period(self, X: NDArray) -> float:
        pass

    @abstractmethod
    def DF(self, *args, **kwargs) -> NDArray:
        pass

    @abstractmethod
    def f(self, *args, **kwargs) -> NDArray:
        pass

    @abstractmethod
    def f_df_stm(self, X: NDArray, *args, **kwargs) -> Tuple[NDArray, NDArray, NDArray]:
        pass


class spatial_perpendicular(Targetter):
    def __init__(self, int_tol: float, mu: float = muEM):
        self.int_tol = int_tol
        self.mu = mu

    def get_X(self, x0: NDArray, period: float):
        return np.array([x0[0], x0[2], x0[-2], period / 2])

    def get_x0(self, X: NDArray):
        return np.array([X[0], 0, X[1], 0, X[2], 0])

    def get_period(self, X: NDArray):
        return X[-1] * 2

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.array(
            [
                [stm[1, 0], stm[1, 2], stm[1, -2], eomf[1]],
                [stm[-3, 0], stm[-3, 2], stm[-3, -2], eomf[-3]],
                [stm[-1, 0], stm[-1, 2], stm[-1, -2], eomf[-1]],
            ]
        )
        return dF

    def f(self, xf: NDArray):
        return np.array([xf[1], xf[-3], xf[-1]])

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = integrate(
            coupled_stm_eom, (0.0, period / 2), xstmIC, self.int_tol, args=(self.mu,)
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(xf)

        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
        return f, dF, stm_full


class fullstate_minus_one(Targetter):
    def __init__(
        self,
        index_fixed: int,
        index_no_enforce: int,
        value_fixed: float,
        int_tol: float,
        mu: float = muEM,
    ):
        self.int_tol = int_tol
        self.mu = mu
        self.ind_fixed = index_fixed
        self.state_val = value_fixed
        self.ind_skip = index_no_enforce
        assert 0 <= self.ind_fixed < 6
        assert 0 <= self.ind_skip < 6

    def get_X(self, x0: NDArray, period: float):
        return np.append(np.delete(x0, self.ind_fixed), period)

    def get_x0(self, X: NDArray):
        states = np.array(X[:-1])
        return np.insert(states, self.ind_fixed, self.state_val)

    def get_period(self, X: NDArray):
        return X[-1]

    def DF(self, stm: NDArray, eomf: NDArray):
        dF = np.hstack((stm - np.eye(6), eomf[:, None]))
        dF = np.delete(dF, self.ind_fixed, 1)
        dF = np.delete(dF, self.ind_skip, 0)
        return dF

    def f(self, x0: NDArray, xf: NDArray):
        state_diff = xf - x0
        out = np.delete(state_diff, self.ind_skip)
        return out

    def f_df_stm(self, X: NDArray):
        x0 = self.get_x0(X)
        period = self.get_period(X)
        xstmIC = np.array([*x0, *np.eye(6).flatten()])
        ts, ys, _, _ = integrate(
            coupled_stm_eom, (0.0, period), xstmIC, self.int_tol, args=(self.mu,)
        )
        xf, stm = ys[:6, -1], ys[6:, -1].reshape(6, 6)
        xf = np.array(xf)
        eomf = eom(ts[-1], xf, self.mu)

        dF = self.DF(stm, eomf)
        f = self.f(x0, xf)
        return f, dF, stm