"""Piecewise Horner spline models for 1D ODE examples.

The spline idea is simple: split the physical domain into intervals and place
one hard-IC Horner polynomial on each interval. Internal boundary values are
propagated from left to right, so continuity is built into the model instead of
being added as a soft penalty.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from Model_Horner import Horner_IC_1_order, Horner_IC_2_order


def _derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Internal derivative helper used for boundary propagation."""

    return torch.autograd.grad(
        y,
        [x],
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


class _SplineBase(nn.Module):
    """Shared validation and tensor helpers for the spline models."""

    def _configure_domain(self, breakpoints, orders):
        self.breakpoints = tuple(float(point) for point in breakpoints)
        self.orders = tuple(int(order) for order in orders)

        if len(self.breakpoints) < 2:
            raise ValueError("breakpoints must contain at least [a, b].")
        if len(self.orders) != len(self.breakpoints) - 1:
            raise ValueError("orders must contain exactly len(breakpoints)-1 values.")

        for left, right in zip(self.breakpoints[:-1], self.breakpoints[1:]):
            if right <= left:
                raise ValueError("breakpoints must be strictly increasing.")

        self.intervals = tuple(zip(self.breakpoints[:-1], self.breakpoints[1:]))

    @property
    def n_intervals(self) -> int:
        return len(self.intervals)

    def _as_column(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t.view(-1, 1)
        if t.ndim != 2 or t.size(1) != 1:
            raise ValueError("t must have shape (N,) or (N, 1).")
        return t

    def _initial_value(self, value, reference: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=reference.device, dtype=reference.dtype).view(1, 1)
        return torch.tensor(value, dtype=reference.dtype, device=reference.device).view(1, 1)


class HornerSplineIC1(_SplineBase):
    """Piecewise Horner model with hard C0 continuity.

    This is used for first-order ODEs. The first interval receives the external
    initial condition y(a)=y0. Every following interval receives its left value
    from the previous interval's right endpoint, so y is continuous at the
    internal breakpoints by construction.
    """

    def __init__(self, breakpoints, orders, initial_value=0.0):
        super().__init__()
        self._configure_domain(breakpoints, orders)
        self.initial_value = initial_value

        for order in self.orders:
            if order < 2:
                raise ValueError("HornerSplineIC1 requires every order >= 2.")

        self.nets = nn.ModuleList(
            Horner_IC_1_order(order, left, right)
            for (left, right), order in zip(self.intervals, self.orders)
        )

    def _resolve_initial_value(self, reference: torch.Tensor, y0=None) -> torch.Tensor:
        return self._initial_value(self.initial_value if y0 is None else y0, reference)

    def _right_boundary(self, interval_idx: int, y_left: torch.Tensor, reference: torch.Tensor):
        _left, right = self.intervals[interval_idx]
        t_right = torch.tensor(right, dtype=reference.dtype, device=reference.device).view(1, 1)
        return self.nets[interval_idx](t_right, y_left)

    def left_value_for_interval(self, interval_idx: int, reference: torch.Tensor, y0=None):
        if interval_idx < 0 or interval_idx >= self.n_intervals:
            raise IndexError("interval_idx is outside the model intervals.")

        y_left = self._resolve_initial_value(reference, y0)
        for idx in range(interval_idx):
            y_left = self._right_boundary(idx, y_left, reference)
        return y_left

    def evaluate_interval(self, interval_idx: int, t: torch.Tensor, y0=None) -> torch.Tensor:
        t = self._as_column(t)
        y_left = self.left_value_for_interval(interval_idx, t, y0)
        return self.nets[interval_idx](t, y_left)

    def forward(self, t: torch.Tensor, y0=None) -> torch.Tensor:
        t = self._as_column(t)
        if t.numel() == 0:
            return torch.empty_like(t)

        y_out = torch.empty_like(t)
        assigned = torch.zeros_like(t, dtype=torch.bool)
        y_left = self._resolve_initial_value(t, y0)

        for idx, (left, right) in enumerate(self.intervals):
            is_last = idx == self.n_intervals - 1
            if is_last:
                mask = torch.logical_and(t >= left, t <= right)
            else:
                mask = torch.logical_and(t >= left, t < right)

            if mask.any():
                values = self.nets[idx](t[mask].view(-1, 1), y_left)
                y_out[mask] = values.view(-1)
                assigned[mask] = True

            if not is_last:
                y_left = self._right_boundary(idx, y_left, t)

        if not bool(assigned.all().item()):
            raise ValueError(f"All t values must be inside [{self.breakpoints[0]}, {self.breakpoints[-1]}].")

        return y_out


class HornerSplineIC2(_SplineBase):
    """Piecewise Horner model with hard C1 continuity.

    This is used for second-order ODEs. The first interval receives y(a)=y0 and
    y'(a)=yd0. Every following interval receives both values from the previous
    interval's right endpoint, so y and y' are continuous at all internal
    breakpoints by construction.
    """

    def __init__(self, breakpoints, orders, initial_conditions=(0.0, 0.0)):
        super().__init__()
        self._configure_domain(breakpoints, orders)
        self.initial_conditions = initial_conditions

        for order in self.orders:
            if order < 3:
                raise ValueError("HornerSplineIC2 requires every order >= 3.")

        self.nets = nn.ModuleList(
            Horner_IC_2_order(order, left, right)
            for (left, right), order in zip(self.intervals, self.orders)
        )

    def _resolve_initial_conditions(self, reference: torch.Tensor, y0=None, yd0=None):
        stored_y0, stored_yd0 = self.initial_conditions
        y0 = stored_y0 if y0 is None else y0
        yd0 = stored_yd0 if yd0 is None else yd0
        return self._initial_value(y0, reference), self._initial_value(yd0, reference)

    def _right_boundary(
        self,
        interval_idx: int,
        y_left: torch.Tensor,
        yd_left: torch.Tensor,
        reference: torch.Tensor,
    ):
        _left, right = self.intervals[interval_idx]
        t_right = torch.tensor(right, dtype=reference.dtype, device=reference.device).view(1, 1)
        t_right = t_right.clone().detach().requires_grad_(True)

        y_right = self.nets[interval_idx](t_right, y_left, yd_left)
        yd_right = _derivative(y_right, t_right)
        return y_right, yd_right

    def left_conditions_for_interval(self, interval_idx: int, reference: torch.Tensor, y0=None, yd0=None):
        if interval_idx < 0 or interval_idx >= self.n_intervals:
            raise IndexError("interval_idx is outside the model intervals.")

        y_left, yd_left = self._resolve_initial_conditions(reference, y0, yd0)
        for idx in range(interval_idx):
            y_left, yd_left = self._right_boundary(idx, y_left, yd_left, reference)
        return y_left, yd_left

    def evaluate_interval(self, interval_idx: int, t: torch.Tensor, y0=None, yd0=None) -> torch.Tensor:
        t = self._as_column(t)
        y_left, yd_left = self.left_conditions_for_interval(interval_idx, t, y0, yd0)
        return self.nets[interval_idx](t, y_left, yd_left)

    def forward(self, t: torch.Tensor, y0=None, yd0=None) -> torch.Tensor:
        t = self._as_column(t)
        if t.numel() == 0:
            return torch.empty_like(t)

        y_out = torch.empty_like(t)
        assigned = torch.zeros_like(t, dtype=torch.bool)
        y_left, yd_left = self._resolve_initial_conditions(t, y0, yd0)

        for idx, (left, right) in enumerate(self.intervals):
            is_last = idx == self.n_intervals - 1
            if is_last:
                mask = torch.logical_and(t >= left, t <= right)
            else:
                mask = torch.logical_and(t >= left, t < right)

            if mask.any():
                values = self.nets[idx](t[mask].view(-1, 1), y_left, yd_left)
                y_out[mask] = values.view(-1)
                assigned[mask] = True

            if not is_last:
                y_left, yd_left = self._right_boundary(idx, y_left, yd_left, t)

        if not bool(assigned.all().item()):
            raise ValueError(f"All t values must be inside [{self.breakpoints[0]}, {self.breakpoints[-1]}].")

        return y_out

