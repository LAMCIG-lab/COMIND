"""
Sensitivity IVPs for d x / d(parameter) using SciPy LSODA.

Forward trajectory x(t) comes from utils.solve_system (LSODA). Sensitivities solve
    u' = J(t,x(t)) u + B(t),
with J = ∂F/∂x for the logistic ODE RHS F.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def _ode_jacobian(x: np.ndarray, K_eff: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Jacobian of dx/dt = (I - diag(x)) @ (K_eff @ x + f)."""
    g = K_eff @ x + f
    J = (1.0 - x)[:, None] * K_eff
    J[np.diag_indices_from(J)] = (1.0 - x) * np.diag(K_eff) - g
    return J


def interp_sensitivity_at_obs(u_traj: np.ndarray, t_span: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """u_traj (n, M), returns (n_q, n)."""
    t_query = np.asarray(t_query, dtype=float)
    n_q = len(t_query)
    n = u_traj.shape[0]
    out = np.zeros((n_q, n))
    for j in range(n):
        out[:, j] = np.interp(t_query, t_span, u_traj[j])
    return out


def integrate_linear_sensitivity_lsoda(
    t_span: np.ndarray,
    x_traj: np.ndarray,
    K_eff: np.ndarray,
    K_mat: np.ndarray,
    f: np.ndarray,
    kind: str,
    index: int | None = None,
) -> np.ndarray:
    """
    Integrate u' = J(t) u + B(t), u(0)=0 on the same grid as t_span.

    kind : 'scalar_K' | 'kappa_diag' | 'f_diag'
    """
    t_span = np.asarray(t_span, dtype=float)
    x_traj = np.asarray(x_traj, dtype=float)
    K_eff = np.asarray(K_eff, dtype=float)
    K_mat = np.asarray(K_mat, dtype=float)
    f = np.asarray(f, dtype=float).ravel()

    n, m = x_traj.shape
    if m != len(t_span):
        raise ValueError("x_traj must have shape (n, len(t_span))")
    if m < 2:
        return np.zeros((n, m))

    x_fn = interp1d(t_span, x_traj, axis=1, kind="linear", fill_value="extrapolate")

    if kind == "scalar_K":
        def forcing(x: np.ndarray) -> np.ndarray:
            return (1.0 - x) * (K_mat @ x)
    elif kind == "kappa_diag":
        b = int(index)

        def forcing(x: np.ndarray) -> np.ndarray:
            out = np.zeros(n)
            out[b] = (1.0 - x[b]) * x[b]
            return out
    elif kind == "f_diag":
        b = int(index)

        def forcing(x: np.ndarray) -> np.ndarray:
            out = np.zeros(n)
            out[b] = 1.0 - x[b]
            return out
    else:
        raise ValueError(f"unknown kind {kind!r}")

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x_fn(t), dtype=float)
        return _ode_jacobian(x, K_eff, f) @ u + forcing(x)

    def jac(t: float, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x_fn(t), dtype=float)
        return _ode_jacobian(x, K_eff, f)

    sol = solve_ivp(
        rhs,
        [float(t_span[0]), float(t_span[-1])],
        np.zeros(n, dtype=float),
        method="LSODA",
        jac=jac,
        t_eval=t_span,
    )
    return sol.y


def integrate_all_sensitivities_lsoda(
    t_span: np.ndarray,
    x_traj: np.ndarray,
    K_eff: np.ndarray,
    K: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """
    Augmented sensitivity IVP for scalar_K and all kappa_b in one LSODA call.

    Solves dU/dt = A(t) @ U + B(t) with U shape (n_biomarkers, n_params),
    n_params = 1 + n_biomarkers. Column 0 is d x / d scalar_K; columns 1..n are
    d x / d kappa_b. U(0) = 0.

    Returns
    -------
    U_traj : (n_biomarkers, n_params, len(t_span))
    """
    t_span = np.asarray(t_span, dtype=float)
    x_traj = np.asarray(x_traj, dtype=float)
    K_eff = np.asarray(K_eff, dtype=float)
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).ravel()

    n, m = x_traj.shape
    n_params = 1 + n
    if m != len(t_span):
        raise ValueError("x_traj must have shape (n, len(t_span))")
    if m < 2:
        return np.zeros((n, n_params, m))

    x_fn = interp1d(t_span, x_traj, axis=1, kind="linear", fill_value="extrapolate")

    def forcing_matrix(x: np.ndarray) -> np.ndarray:
        B = np.zeros((n, n_params), dtype=float)
        B[:, 0] = (1.0 - x) * (K @ x)
        for b in range(n):
            B[b, b + 1] = (1.0 - x[b]) * x[b]
        return B

    def rhs(t: float, u_flat: np.ndarray) -> np.ndarray:
        U = u_flat.reshape(n, n_params, order="F")
        x = np.asarray(x_fn(t), dtype=float)
        A = _ode_jacobian(x, K_eff, f)
        dU = A @ U + forcing_matrix(x)
        return dU.ravel(order="F")

    def jac(t: float, u_flat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_fn(t), dtype=float)
        A = _ode_jacobian(x, K_eff, f)
        return np.kron(np.eye(n_params), A)

    sol = solve_ivp(
        rhs,
        [float(t_span[0]), float(t_span[-1])],
        np.zeros(n * n_params, dtype=float),
        method="LSODA",
        jac=jac,
        t_eval=t_span,
    )
    return sol.y.reshape(n, n_params, m, order="F")


def integrate_f_sensitivities_lsoda(
    t_span: np.ndarray,
    x_traj: np.ndarray,
    K_eff: np.ndarray,
    K: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """
    Augmented sensitivity IVP for all forcing terms f_b in one LSODA call.

    Solves dU/dt = A(t) @ U + B(t) with U shape (n_biomarkers, n_biomarkers).
    Column b is d x / d f_b. B(t) = diag(1 - x(t)). U(0) = 0.

    Returns
    -------
    U_traj : (n_biomarkers, n_biomarkers, len(t_span))
    """
    t_span = np.asarray(t_span, dtype=float)
    x_traj = np.asarray(x_traj, dtype=float)
    K_eff = np.asarray(K_eff, dtype=float)
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).ravel()

    n, m = x_traj.shape
    n_params = n
    if m != len(t_span):
        raise ValueError("x_traj must have shape (n, len(t_span))")
    if m < 2:
        return np.zeros((n, n_params, m))

    x_fn = interp1d(t_span, x_traj, axis=1, kind="linear", fill_value="extrapolate")

    def forcing_matrix(x: np.ndarray) -> np.ndarray:
        return np.diag(1.0 - x)

    def rhs(t: float, u_flat: np.ndarray) -> np.ndarray:
        U = u_flat.reshape(n, n_params, order="F")
        x = np.asarray(x_fn(t), dtype=float)
        A = _ode_jacobian(x, K_eff, f)
        dU = A @ U + forcing_matrix(x)
        return dU.ravel(order="F")

    def jac(t: float, u_flat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_fn(t), dtype=float)
        A = _ode_jacobian(x, K_eff, f)
        return np.kron(np.eye(n_params), A)

    sol = solve_ivp(
        rhs,
        [float(t_span[0]), float(t_span[-1])],
        np.zeros(n * n_params, dtype=float),
        method="LSODA",
        jac=jac,
        t_eval=t_span,
    )
    return sol.y.reshape(n, n_params, m, order="F")
