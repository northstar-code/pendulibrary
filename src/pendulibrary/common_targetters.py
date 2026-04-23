import numpy as np
from pendulibrary.integrate import integrate_state_stm, eom
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


class Targetter(ABC):
    @abstractmethod
    def __init__(self, Lr: float, Mr: float, int_tol: float, *args, **kwargs):
        self.Mr = np.nan
        self.Lr = np.nan
        self.int_tol = np.nan

    @abstractmethod
    def get_X(self, x0: np.ndarray, period: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_x0(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_period(self, X: np.ndarray) -> float:
        pass

    @abstractmethod
    def DG(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def g(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def g_dg_stm(
        self, X: np.ndarray, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class single_fixed(Targetter):
    def __init__(
        self,
        ind_fixed: int,
        val_fixed: float,
        ind_no_enforce: int,
        Lr: float,
        Mr: float,
        int_tol: float,
    ):
        self.int_tol = int_tol
        self.Lr = Lr
        self.Mr = Mr
        self.val_fixed = val_fixed
        self.ind_fixed = ind_fixed
        self.ind_no_enforce = ind_no_enforce

    def get_X(self, x0: np.ndarray, period: float):
        return np.append(np.delete(x0, self.ind_fixed), period)

    def get_x0(self, X: np.ndarray):
        states = np.array(X[:-1])
        return np.insert(states, self.ind_fixed, self.val_fixed)

    def get_period(self, X: np.ndarray):
        return float(X[-1])

    def DG(self, stm: np.ndarray, eomf: np.ndarray):
        dG = np.hstack((stm - np.eye(4), eomf[:, None]))
        dG = np.delete(dG, self.ind_fixed, 1)
        dG = np.delete(dG, self.ind_no_enforce, 0)
        return dG

    def g(self, x0: np.ndarray, xf: np.ndarray):
        return np.delete(xf - x0, self.ind_no_enforce)

    def g_dg_stm(self, X: np.ndarray):
        x0 = self.get_x0(X)
        period = self.get_period(X)
        _, ys = integrate_state_stm(x0, period, self.Lr, self.Mr, self.int_tol)
        xf, stm = ys[:4, -1].copy(), ys[4:, -1].reshape(4, 4)
        eomf = eom(0.0, xf, self.Lr, self.Mr)

        dG = self.DG(stm, eomf)
        g = self.g(x0, xf)

        return g, dG, stm
