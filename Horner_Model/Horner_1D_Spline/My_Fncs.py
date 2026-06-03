"""Shared helper functions for the 1D Horner spline examples."""

from __future__ import annotations

import torch


def derivative(y: torch.Tensor, x: torch.Tensor, grad_outputs=None) -> torch.Tensor:
    """Compute dy/dx using PyTorch autograd."""

    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True to calculate derivatives.")

    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)

    return torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True,
    )[0]


def count_parameters(model, verbose: bool = True) -> int:
    """Count trainable model parameters."""

    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if verbose:
        print("Number of parameters:")
        print(count)
        print("\n")
    return count


def print_all_parameters(model) -> None:
    """Print every named trainable parameter for small-model diagnostics."""

    for name, parameter in model.named_parameters():
        print(f"parameter name: {name}")
        print(parameter)
        print("\n")


def select_subdomain(t: torch.Tensor, u: torch.Tensor, left: float, right: float, is_last: bool):
    """Select collocation points that belong to one spline interval."""

    if is_last:
        mask = torch.logical_and(t >= left, t <= right)
    else:
        mask = torch.logical_and(t >= left, t < right)

    return t[mask].view(-1, 1), u[mask].view(-1, 1)


def evaluate_piecewise_solution(t_plot_tensor: torch.Tensor, model):
    """Evaluate y, y', and y'' interval by interval on a sorted plotting grid."""

    y_parts = []
    yd_parts = []
    ydd_parts = []

    for interval_idx, (left, right) in enumerate(model.intervals):
        is_last = interval_idx == model.n_intervals - 1
        if is_last:
            mask = torch.logical_and(t_plot_tensor >= left, t_plot_tensor <= right)
        else:
            mask = torch.logical_and(t_plot_tensor >= left, t_plot_tensor < right)

        if not mask.any():
            continue

        t_sub = t_plot_tensor[mask].view(-1, 1)
        y = model.evaluate_interval(interval_idx, t_sub)
        yd = derivative(y, t_sub)
        ydd = derivative(yd, t_sub)

        y_parts.append(y)
        yd_parts.append(yd)
        ydd_parts.append(ydd)

    return torch.cat(y_parts), torch.cat(yd_parts), torch.cat(ydd_parts)


def rmse_error(y, y_exact) -> float:
    """Compute the standard RMSE error on the plotting grid."""

    return float(((y - y_exact) ** 2).mean() ** 0.5)


def should_stop(train_error, epoch: int, tolerance: float = 1e-16) -> bool:
    """Return True after three nearly identical consecutive losses."""

    if epoch < 2:
        return False

    return (
        abs(train_error[epoch] - train_error[epoch - 1]) < tolerance
        and abs(train_error[epoch - 1] - train_error[epoch - 2]) < tolerance
    )

