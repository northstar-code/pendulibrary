from pendulibrary.targeter import dc_square, dc_tangent, dc_underconstrained
from warnings import warn
from typing import List, Callable
from tqdm.auto import tqdm
import numpy as np


def fixed_step_cont(
    X0: np.ndarray,
    g_dg_stm_func: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dir0: np.ndarray | List,
    s: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    max_iter: None | int = None,
    max_step: None | float = None,
    fudge: float | None = None,
    exact_tangent: bool = False,
) -> tuple[List, List]:
    """Custom arclength-based continuation wrapper. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X0 (np.ndarray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (np.ndarray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        S (float, optional): terminate at this arclength. Defaults to 0.5.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        max_iter: (int | None, optional): maximum number of iterations. Will return what it's computed so far if it exceeds that
        fudge: (float | None, optional): multiply step size by this much in the differential corrector
        exact_tangent (bool, optional): whether the tangent vector `dir0` passed in is exact or approximate. If approximate, it is only used to check direction with a dot product. Otherwise, it is used as-is.
        modified (bool, optional): Whether to use modified algorithm. Defaults to True.
        stop_callback (Callable): Function with signature f(X, current_eigvals, previous_eigvals, *kwargs) which returns True when continuation should terminate. If None, will only terminate when the final arclength is reached. Defaults to None.
        stop_kwags (dict, optional): keyword arguments to stop_calback. Defaults to {}.


    Returns:
        tuple[List, List]: all Xs, all eigenvalues
    """
    # if no stop callback, make one

    X = X0.copy()
    tangent_prev = dir0 / np.linalg.norm(dir0)

    _, dF, stm = g_dg_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = dir0.copy() if exact_tangent else svd.Vh[-1]

    # # if the direction we asked for is normal to the computed tangent, use the second-most tangent vector
    # if np.abs(np.dot(tangent, dir0)) < 1e-5:
    #     print("RESETTING")
    #     tangent = svd.Vh[-1]

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=S)
    arclen = 0.0

    # ensure that the stopping condition hasnt been satisfied
    while arclen < S:
        # if we flip flop, undo the flipflop
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        try:
            X, dF, stm, _ = dc_tangent(
                X,
                tangent,
                g_dg_stm_func,
                s,
                tol,
                max_iter=max_iter,
                max_step=max_step,
                fudge=fudge,
            )
        except np.linalg.LinAlgError as err:
            print(f"Linear algebra error encountered: {err}")
            print("returning what's been calculated so far")
            break
        except RuntimeError as err:
            print(f"Runtime error encountered: {err}")
            print("returning what's been calculated so far")
            break

        Xs.append(X)

        eig_vals.append(np.linalg.eigvals(stm))
        dS = s

        tangent_prev = tangent

        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]

        arclen += dS
        bar.update(float(dS))

    bar.close()

    return Xs, eig_vals


def adaptive_cont(
    X0: np.ndarray,
    g_dg_stm_func: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dir0: np.ndarray | List,
    s0: float = 1e-2,
    s_min: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 10,
    target_iter: int = 6,
    rate: float = 1.15,
    reduce_maxiter: float = 5.0,
    reduce_reverse: float = 2.0,
    exp_direction: float = 10.0,
    exp_iters: float = 0.3,
    max_step: float | None = None,
    exact_tangent: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Custom arclength-based continuation wrapper with variable step size. This modified algorithm has a full step size of s, rather than projected step size.
    At each step, the step size multiplies by num_iters/num_iters_previous, in so that if it takes longer to converge we reduce the step size
    At each step, the step size also multiplies by the dot product between the tangent vector and the step; if this dot product is close to 1, then the curve is not sharp and step size wont be reduced. Else, it will.
    At each step, the step size also multiplies by the parameter `rate`, which should be >1 to ensure step size can recover

    Args:
        X0 (np.ndarray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (np.ndarray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        S (float, optional): terminate at this arclength. Defaults to 0.5.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        max_iter: (int | None, optional): maximum number of iterations. Will return what it's computed so far if it exceeds that
        rate (float, optional): the rate of increase of step size in the absense of any other change
        reduce_maxiter (float, optional): If we hit the maximum iterations on one attempt, reduce the step size by a factor of this much
        reduce_reverse (float, optional): If there's a possibility that the solution curve is reversing backward on itself, reduce the step size by a factor of this much
        exact_tangent (bool, optional): whether the tangent vector `dir0` passed in is exact or approximate. If approximate, it is only used to check direction with a dot product. Otherwise, it is used as-is.


    Returns:
        tuple[List, List]: all Xs, all eigenvalues
    """
    assert rate >= 1.0
    assert reduce_maxiter > 1.0
    assert reduce_reverse > 1.0
    assert max_iter > target_iter

    X = X0.copy()
    tangent_prev = dir0 / np.linalg.norm(dir0)

    _, dF, stm = g_dg_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = tangent_prev.copy() if exact_tangent else svd.Vh[-1]

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]
    tangents = [tangent.copy()]
    DFs = [dF]
    stms = [stm]
    # s_vals = [0.0]

    bar = tqdm(total=S)
    arclen = 0.0
    s = s0

    niters = 0

    close_to_start = False

    try:
        while arclen < S:
            bar.set_description(f"s: {s:.3e}")
            try:
                X, dF, stm, niters = dc_tangent(
                    X,
                    tangent,
                    g_dg_stm_func,
                    s,
                    tol,
                    max_iter=max_iter,
                    max_step=s,
                    debug=False,
                )
            except KeyboardInterrupt as err:
                bar.set_postfix_str("KeyboardInterrupt, premature termination")
                bar.close()
                break
            except RuntimeError as err:
                if "max iterations" in str(err):
                    s /= reduce_maxiter
                    continue
                else:
                    raise err

            dprod_check = np.dot(tangent, X - Xs[-1]) / s
            if dprod_check < 0.25:
                # reject the last step
                s /= reduce_reverse
                dS = np.linalg.norm(Xs[-1] - Xs[-2])
                arclen -= dS
                bar.update(-dS)
                Xs.pop()
                tangents.pop()
                DFs.pop()
                # s_vals.pop()
                X = Xs[-1]
                tangent = tangent_prev.copy()
                continue

            Xs.append(X)
            tangents.append(tangent)
            eig_vals.append(np.linalg.eigvals(stm))
            DFs.append(dF)
            stms.append(stm)
            arclen += s

            tangent_prev = tangent

            svd = np.linalg.svd(dF)
            tangent = svd.Vh[-1]
            # if we flip flop, undo the flipflop
            if np.dot(tangent, Xs[-1] - Xs[-2]) < 0:
                tangent *= -1

            bar.update(float(s))

            # looped when youre not near the beginning and within $s$ of the initial state
            Xdif = X - X0
            close_to_start = (
                np.sum((Xdif) ** 2) < (s * 1.5) ** 2 and np.dot(Xdif, tangents[0]) > 0.0
            )

            if arclen > 5 * s and close_to_start:
                bar.set_postfix_str("Completed loop")
                bar.total = arclen
                bar.colour = "green"
                bar.set_description(f"mean ds: {arclen/len(Xs):.3e}")
                bar.refresh()
                break

            s *= (target_iter / niters) ** exp_iters * dprod_check**exp_direction
            if max_step is not None and s > max_step:
                s = max_step

            if niters <= target_iter:
                s *= rate

            if s < s_min:
                bar.set_postfix_str("Stepsize smaller than min, premature termination")
                bar.colour = "red"
                bar.refresh()
                close_to_start = False
                break

        if not close_to_start and arclen >= S:  # if we reach the arclength max
            bar.set_postfix_str("Reached max arclength")
    except BaseException as err:
        err_name = type(err).__name__
        err_text = str(err)
        bar.set_postfix_str(f"{err_name}: {err_text}, premature termination")
    bar.close()

    return (
        np.array(Xs),
        np.array(eig_vals),
        (np.array(DFs), np.array(tangents), np.array(stms)),
    )


def find_bifurcation(
    X0: np.ndarray | List,
    g_dg_stm_func: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    tan0: np.ndarray | List,
    s0: float = 1e-3,
    targ_tol: float = 1e-13,
    bisect_tol: float = 1e-5,
    period_mult: float = 1,
    debug: bool = False,
    scale: float = 5,
    seek_local_opt: bool = False,
) -> np.ndarray:
    """Find bifurcation using Broucke stability

    Args:
        X0 (np.ndarray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (np.ndarray | List): signed initial stepoff direction.
        s0 (float, optional): initial step size. Defaults to 1e-2.
        targ_tol (float, optional): tolerance for targetter convergence. Defaults to 1e-10.
        skip (int, optional): number of crossings to skip. Defaults to 0.
        bisect_tol (float, optional): Tolerance for bisection algorithm. Defaults to 1e-5.
        bif_type (str, optional): bif_type of bifurcation to detect ("tangent", "hopf") OR
            a tuple indicating period-multiplying bifurcation (e.g. (3,)
            for tripling, (5,2) for quintupling with second harmonic). Defaults to "tangent".
        debug (bool, optional): whether to print off function evaluations and steps

    Returns:
        np.ndarray: Bifurcation control variables, tangent vector
    """
    if seek_local_opt:
        assert scale > 4
    nu_target = 2 * np.cos(2 * np.pi / period_mult)
    bisect_func = lambda eigs: np.sum(eigs).real - 2 - nu_target

    X = np.array(X0) if isinstance(X0, list) else X0.copy()
    tangent_prev = tan0.copy()
    s = s0

    _, dG, stm = g_dg_stm_func(X)
    svd = np.linalg.svd(dG)
    tangent = svd.Vh[-1]

    Xs = [X0]
    eigs = np.linalg.eigvals(stm)

    func_vals = [bisect_func(eigs)]

    switch_loc = False
    counter = 0
    while True:
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        X, dG, stm, _ = dc_tangent(
            X, np.sign(s) * tangent, g_dg_stm_func, abs(s), targ_tol, max_step=abs(s)
        )
        counter += 1
        Xs.append(X.copy())
        tangent_prev = tangent.copy()

        svd = np.linalg.svd(dG)
        tangent = svd.Vh[-1]

        func_vals.append(bisect_func(np.linalg.eigvals(stm)))

        # if we have >3 vals and none of the last 3 are nan and the dif from the previous two is different
        local_opt = (
            len(func_vals) >= 3
            and (not np.any(np.isnan(func_vals[-3:])))
            and (
                np.sign(func_vals[-1] - func_vals[-2])
                != np.sign(func_vals[-2] - func_vals[-3])
            )
        )
        if seek_local_opt and local_opt:
            switch_loc = True

        if abs(func_vals[-1]) < bisect_tol or abs(s) < bisect_tol:
            X = Xs[np.nanargmin(np.abs(func_vals))]
            return X
        if (not np.any(np.isnan(func_vals[-3:]))) and (
            (np.sign(func_vals[-1]) != np.sign(func_vals[-2])) or (switch_loc)
        ):
            # if abs(func_vals[-1]) < bisect_tol or abs(s) < bisect_tol:
            #     X = Xs[np.nanargmin(np.abs(func_vals))]
            #     return X
            # else:  # search backward
            s /= -scale
            if switch_loc:
                switch_loc = False
                Xs.append(np.nan)
                func_vals.append(np.nan)

        if debug:
            print(func_vals[-1], func_vals[-2], s)
