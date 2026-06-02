"""Polynomial regression for a second-order ODE example.

This script is the Python equivalent of ``ODE_order2.m``.

It approximates the solution of:

    x''(t) + 2*x'(t) + x(t) = sin(t),     x(0) = 0, x'(0) = 0

with a polynomial in scaled Taylor form:

    P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from polynomial_ode_utils import (
    evaluate_scaled_polynomial,
    plot_model_and_error,
    print_run_summary,
    rmse_error,
    save_or_show_figures,
    scaled_power_matrix,
)


def parse_args() -> argparse.Namespace:
    """Read optional command-line settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatable collocation points.")
    parser.add_argument("--no-show", action="store_true", help="Create figures without opening a plot window.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory where PNG figures should be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Main parameters:
    # n = order of the differential equation. Here n = 2, so two initial
    #     conditions are fixed: P(0) = c0 and P'(0) = c1.
    # m = degree of the polynomial approximation. Here m = 15, so the model
    #     uses coefficients c0, c1, ..., c15.
    # M = number of random collocation points. The ODE is sampled at these
    #     points, giving an overdetermined linear system solved by least squares.
    # In this script, n is mainly a readable reminder of the ODE order.
    n = 2
    m = 15
    M = 10_000
    _ = n

    rng = np.random.default_rng(args.seed)

    # Random collocation points. This matches the MATLAB expression
    # tk = 6.1*rand(M,1)-0.05.
    tk = 6.1 * rng.random(M) - 0.05

    # Right-hand side of the ODE.
    uk0 = np.sin(tk)

    # Initial conditions fix the first two polynomial coefficients:
    # c0 = P(0), c1 = P'(0).
    c0 = 0.0
    c1 = 0.0

    # Move the known contributions of c0 and c1 to the right-hand side.
    # The formula is written in the same form as the MATLAB script, even
    # though c0 and c1 are zero in this example.
    uk = uk0 - 1.0 * (c0 + c1 * tk) - 2.0 * c1

    # T1 represents the second derivative part P''(t), using c2, ..., cm.
    T1 = scaled_power_matrix(tk, range(0, m - 1))

    # T2 represents the first derivative part P'(t), using c2, ..., cm.
    T2 = scaled_power_matrix(tk, range(1, m))

    # T3 represents the polynomial part P(t), excluding fixed c0 and c1 terms.
    T3 = scaled_power_matrix(tk, range(2, m + 1))

    # Linear system for the unknown coefficients:
    #
    #     P''(tk) + 2*P'(tk) + P(tk) = sin(tk)
    #
    # at all sampled collocation points tk.
    A = T1 + 2.0 * T2 + T3

    # Solve the overdetermined linear system in the least-squares sense.
    ch = np.linalg.pinv(A) @ uk

    # Full coefficient vector, including the fixed initial-condition terms.
    coefficients = np.concatenate(([c0, c1], ch))

    # Dense grid used only for plotting and measuring the RMSE error.
    t = np.linspace(0.0, 6.0, 10_000)

    # Evaluate the polynomial model P(t).
    P = evaluate_scaled_polynomial(coefficients, t, derivative_order=0)

    # Exact solution of the second-order ODE with the chosen initial conditions.
    x = (np.exp(-t) + t * np.exp(-t) - np.cos(t)) / 2.0

    # RMSE error is a single number that summarizes the average size of
    # P(t)-x(t) over the plotting grid.
    solution_rmse = rmse_error(P, x)
    print_run_summary("ODE_order2", coefficients, solution_rmse)

    solution_figure = plot_model_and_error(
        t,
        P,
        x,
        title="Polynomial model solution and exact solution",
        y_label="P, x",
        error_label="P - x",
        x_limits=(0.0, 6.0),
        y_limits=(-0.5, 0.65),
        legend_location="upper right",
    )

    # Evaluate the first derivative P'(t).
    Pd = evaluate_scaled_polynomial(coefficients, t, derivative_order=1)

    # Exact first derivative of the exact solution.
    xd = (np.sin(t) - t * np.exp(-t)) / 2.0

    derivative_figure = plot_model_and_error(
        t,
        Pd,
        xd,
        title="Polynomial model solution and exact solution - derivative",
        y_label="P', x'",
        error_label="P' - x'",
        x_limits=(0.0, 6.0),
        y_limits=(-0.6, 0.4),
        legend_location="upper right",
    )

    # Evaluate the second derivative P''(t).
    Pdd = evaluate_scaled_polynomial(coefficients, t, derivative_order=2)

    # Exact second derivative of the exact solution.
    xdd = (t * np.exp(-t) - np.exp(-t) + np.cos(t)) / 2.0

    second_derivative_figure = plot_model_and_error(
        t,
        Pdd,
        xdd,
        title="Polynomial model solution and exact solution - second derivative",
        y_label="P'', x''",
        error_label="P'' - x''",
        x_limits=(0.0, 6.0),
        y_limits=(-0.5, 0.501),
        legend_location="lower right",
    )

    save_or_show_figures(
        [solution_figure, derivative_figure, second_derivative_figure],
        show=not args.no_show,
        save_dir=args.save_dir,
        filename_prefix="ODE_order2",
    )


if __name__ == "__main__":
    main()
