"""
Sensitivity IVPs for d x / d(parameter) via fixed-step RK4 on the forward grid.

Uses the trajectory x(t) already computed by solve_system; does not re-integrate x.
"""
from __future__ import annotations

import numba
import numpy as np


def _precompute_A_all(x_traj: np.ndarray, K_eff: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Build A(t) for all T timesteps: shape (T, n, n)."""
    n, T = x_traj.shape
    Kx_f = K_eff @ x_traj + f[:, None]  # (n, T)
    A = (1.0 - x_traj).T[:, :, None] * K_eff[None]  # (T, n, n)
    A[:, np.arange(n), np.arange(n)] -= Kx_f.T
    return A


def _precompute_B_all(
    x_traj: np.ndarray, K: np.ndarray, param_type: str
) -> np.ndarray:
    """Build forcing matrix B(t) for all T timesteps."""
    n, T = x_traj.shape
    if param_type == "globals":
        B = np.zeros((T, n, n + 1))
        B[:, :, 0] = ((1.0 - x_traj) * (K @ x_traj)).T
        B[:, np.arange(n), np.arange(n) + 1] = ((1.0 - x_traj) * x_traj).T
    elif param_type == "f":
        B = np.zeros((T, n, n))
        B[:, np.arange(n), np.arange(n)] = (1.0 - x_traj).T
    else:
        raise ValueError(f"param_type must be 'globals' or 'f', got {param_type!r}")
    return B


@numba.njit(cache=True)
def _rk4_loop_numba(t_span, A_all, B_all, n, n_params, T):
    U = np.zeros((n, n_params))
    U_traj = np.zeros((T, n, n_params))
    for i in range(T - 1):
        dt = t_span[i + 1] - t_span[i]
        Ai = A_all[i]
        Ai1 = A_all[i + 1]
        Bi = B_all[i]
        Bi1 = B_all[i + 1]
        Amid = 0.5 * (Ai + Ai1)
        Bmid = 0.5 * (Bi + Bi1)

        k1 = Ai @ U + Bi
        k2 = Amid @ (U + 0.5 * dt * k1) + Bmid
        k3 = Amid @ (U + 0.5 * dt * k2) + Bmid
        k4 = Ai1 @ (U + dt * k3) + Bi1
        U = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        U_traj[i + 1] = U
    return U_traj


def integrate_all_sensitivities_rk4(
    t_span: np.ndarray,
    x_traj: np.ndarray,
    K_eff: np.ndarray,
    K: np.ndarray,
    f: np.ndarray,
    param_type: str,
) -> np.ndarray:
    """
    RK4 integration of dU/dt = A(x(t)) @ U + B(x(t)), U(0) = 0 on t_span.

    Parameters
    ----------
    t_span : (T,) time grid (same as solve_system)
    x_traj : (n_biomarkers, T) forward state trajectory
    K_eff, K, f : ODE parameters (same as forward solve)
    param_type : 'globals' or 'f'

    Returns
    -------
    U_traj : (n_biomarkers, n_params, T)
        'globals': n_params = n + 1; col 0 = d x / d scalar_K, cols 1..n = d x / d kappa_b
        'f': n_params = n; col b = d x / d f_b
    """
    t_span = np.asarray(t_span, dtype=float)
    x_traj = np.asarray(x_traj, dtype=float)
    K_eff = np.asarray(K_eff, dtype=float)
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).ravel()

    n, T = x_traj.shape
    if T != len(t_span):
        raise ValueError("x_traj must have shape (n, len(t_span))")
    if param_type == "globals":
        n_params = n + 1
    elif param_type == "f":
        n_params = n
    else:
        raise ValueError(f"param_type must be 'globals' or 'f', got {param_type!r}")

    if T < 2:
        return np.zeros((n, n_params, T))

    A_all = _precompute_A_all(x_traj, K_eff, f)
    B_all = _precompute_B_all(x_traj, K, param_type)

    U_traj = _rk4_loop_numba(t_span, A_all, B_all, n, n_params, T)
    return np.moveaxis(U_traj, 0, -1)
