"""Shared helpers for polynomial ODE regression examples.

The three example scripts all use the same polynomial basis:

    P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!

This module keeps the repeated linear algebra and plotting code in one place
so the individual ODE scripts can focus on the mathematical problem they are
solving.
"""

from __future__ import annotations

from math import factorial
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def scaled_power_matrix(t: np.ndarray, degrees: Iterable[int]) -> np.ndarray:
    """Return columns t^k/k! for the requested polynomial degrees.

    Parameters
    ----------
    t:
        One-dimensional array of evaluation points.
    degrees:
        Polynomial degrees to include as columns.

    Returns
    -------
    np.ndarray
        Matrix whose j-th column is t**degrees[j] / degrees[j]!.
    """

    t = np.asarray(t, dtype=float).reshape(-1)
    degrees = np.asarray(list(degrees), dtype=int)
    factorials = np.array([factorial(int(k)) for k in degrees], dtype=float)

    return t[:, None] ** degrees[None, :] / factorials[None, :]


def evaluate_scaled_polynomial(
    coefficients: np.ndarray,
    t: np.ndarray,
    derivative_order: int = 0,
) -> np.ndarray:
    """Evaluate P(t) or one of its derivatives.

    The coefficient vector is interpreted as:

        coefficients[i] = ci in ci*t^i/i!

    Because of the factorial scaling, differentiating simply shifts the
    coefficient index. For example:

        P'(t) = c1 + c2*t/1! + c3*t^2/2! + ...

    Parameters
    ----------
    coefficients:
        Polynomial coefficients c0, c1, ..., cm.
    t:
        Evaluation points.
    derivative_order:
        0 for P(t), 1 for P'(t), 2 for P''(t), and so on.
    """

    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative")

    if derivative_order >= len(coefficients):
        return np.zeros_like(t)

    shifted_coefficients = coefficients[derivative_order:]
    degrees = range(len(shifted_coefficients))
    basis = scaled_power_matrix(t, degrees)

    return basis @ shifted_coefficients


def rmse_error(model_values: np.ndarray, exact_values: np.ndarray) -> float:
    """Compute the RMSE error between model values and exact values.

    RMSE means root mean squared error:

        sqrt(mean((model_values - exact_values)^2))

    It is a single number that summarizes the average size of the pointwise
    error on the plotting grid.
    """

    model_values = np.asarray(model_values, dtype=float)
    exact_values = np.asarray(exact_values, dtype=float)

    return float(np.sqrt(np.mean((model_values - exact_values) ** 2)))


def print_run_summary(name: str, coefficients: np.ndarray, rmse: float) -> None:
    """Print a compact numerical summary after fitting the polynomial."""

    np.set_printoptions(precision=10, suppress=False)
    print(f"\n{name}")
    print("-" * len(name))
    print("Polynomial coefficients c0, c1, ..., cm:")
    print(coefficients)
    print(f"RMSE error on plotting grid: {rmse:.10e}")


def plot_model_and_error(
    t: np.ndarray,
    model_values: np.ndarray,
    exact_values: np.ndarray,
    *,
    title: str,
    y_label: str,
    error_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float] | None,
    legend_location: str,
) -> plt.Figure:
    """Create the two-panel plot used by each MATLAB example.

    The top panel compares the polynomial model with the exact solution. The
    bottom panel shows the pointwise error, not the scalar RMSE error.
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(title, fontsize=18)

    axes[0].plot(t, model_values, "-", linewidth=3, color=(0.0, 0.0, 1.0), label="Polynomial model")
    axes[0].plot(t, exact_values, "--", linewidth=4, color=(1.0, 0.0, 0.0), label="Exact solution")
    axes[0].legend(loc=legend_location, fontsize=12)
    axes[0].set_xlabel("t", fontsize=14)
    axes[0].set_ylabel(y_label, fontsize=14)
    axes[0].set_xlim(x_limits)
    if y_limits is not None:
        axes[0].set_ylim(y_limits)
    axes[0].grid(True)

    axes[1].plot(t, model_values - exact_values, "-", linewidth=3, color=(0.0, 0.0, 1.0))
    axes[1].set_title("Pointwise error", fontsize=16)
    axes[1].set_xlabel("t", fontsize=14)
    axes[1].set_ylabel(error_label, fontsize=14)
    axes[1].set_xlim(x_limits)
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def save_or_show_figures(
    figures: list[plt.Figure],
    *,
    show: bool,
    save_dir: Path | None,
    filename_prefix: str,
) -> None:
    """Save figures, show them interactively, or both."""

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, figure in enumerate(figures, start=1):
            figure.savefig(save_dir / f"{filename_prefix}_figure_{index}.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close("all")
