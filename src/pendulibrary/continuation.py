from pendulibrary.targeter import *
from warnings import warn
from typing import List, Callable
from tqdm.auto import tqdm


def fixed_step_cont(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    max_iter: None | int = None,
    max_step: None | float = None,
    fudge: float | None = None,
    exact_tangent: bool = False,
) -> Tuple[List, List]:
    """Custom arclength-based continuation wrapper. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
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
        Tuple[List, List]: all Xs, all eigenvalues
    """
    # if no stop callback, make one

    X = X0.copy()
    tangent_prev = dir0 / np.linalg.norm(dir0)

    _, dF, stm = f_df_stm_func(X0)
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
                f_df_stm_func,
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
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
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
    exact_tangent: bool = False,
) -> Tuple[List, List, Tuple[List, List]]:
    """Custom arclength-based continuation wrapper with variable step size. This modified algorithm has a full step size of s, rather than projected step size.
    At each step, the step size multiplies by num_iters/num_iters_previous, in so that if it takes longer to converge we reduce the step size
    At each step, the step size also multiplies by the dot product between the tangent vector and the step; if this dot product is close to 1, then the curve is not sharp and step size wont be reduced. Else, it will.
    At each step, the step size also multiplies by the parameter `rate`, which should be >1 to ensure step size can recover

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        S (float, optional): terminate at this arclength. Defaults to 0.5.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        max_iter: (int | None, optional): maximum number of iterations. Will return what it's computed so far if it exceeds that
        rate (float, optional): the rate of increase of step size in the absense of any other change
        reduce_maxiter (float, optional): If we hit the maximum iterations on one attempt, reduce the step size by a factor of this much
        reduce_reverse (float, optional): If there's a possibility that the solution curve is reversing backward on itself, reduce the step size by a factor of this much
        exact_tangent (bool, optional): whether the tangent vector `dir0` passed in is exact or approximate. If approximate, it is only used to check direction with a dot product. Otherwise, it is used as-is.


    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    assert rate >= 1.0
    assert reduce_maxiter > 1.0
    assert reduce_reverse > 1.0
    assert max_iter > target_iter

    X = X0.copy()
    tangent_prev = dir0 / np.linalg.norm(dir0)

    _, dF, stm = f_df_stm_func(X0)
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
                    f_df_stm_func,
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
