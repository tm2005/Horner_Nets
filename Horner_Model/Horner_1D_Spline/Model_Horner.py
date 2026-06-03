"""Base Horner-polynomial models used by the 1D spline examples.

These classes represent scalar functions y(t) with polynomials evaluated by
Horner's scheme. The hard-IC variants embed initial conditions directly in the
forward pass, so the training scripts can focus on the ODE residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_parameter_matrix(
    in_features: int,
    out_features: int,
    rand_dist_range: float = 0.05,
    rand_dist_mean: float = 0.0,
) -> nn.Parameter:
    """Create one learnable coefficient matrix for the custom layers."""

    values = (
        2.0
        * rand_dist_range
        * torch.rand(out_features, in_features, dtype=torch.float32).view(
            out_features, in_features
        )
        - rand_dist_range
        + rand_dist_mean
    )
    return nn.Parameter(values, requires_grad=True)


def _constant_like(value: float, reference: torch.Tensor) -> torch.Tensor:
    """Create a scalar tensor on the same device and dtype as `reference`."""

    return torch.full((1, 1), value, dtype=reference.dtype, device=reference.device)


def _scaled_to_minus_one_one(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Scale physical coordinates from [a, b] to the polynomial interval [-1, 1]."""

    return 2.0 / (b - a) * x - (b + a) / (b - a)


class myBias1(nn.Module):
    """Small learnable additive coefficient layer.

    The layer computes

        output = input + b * out_scale

    where `b` is trainable. It is used as one coefficient in the Horner
    recurrence.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_scale: float = 1.0,
        rand_dist_range: float = 0.05,
        rand_dist_mean: float = 0.0,
    ):
        super().__init__()
        self.b = _init_parameter_matrix(
            in_features, out_features, rand_dist_range, rand_dist_mean
        )
        self.out_scale = out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b * self.out_scale


class myLinear1(nn.Module):
    """Small learnable multiplicative coefficient layer.

    In these examples the input and output are scalar, so this is effectively a
    trainable scalar multiplication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_scale: float = 1.0,
        rand_dist_range: float = 0.05,
        rand_dist_mean: float = 0.0,
    ):
        super().__init__()
        self.a = _init_parameter_matrix(
            in_features, out_features, rand_dist_range, rand_dist_mean
        )
        self.out_scale = out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.a) * self.out_scale


class Horner(nn.Module):
    """Plain polynomial model evaluated by Horner's scheme."""

    def __init__(self, order: int, a: float, b: float):
        super().__init__()
        if order < 1:
            raise ValueError("order must be at least 1.")

        self.a = float(a)
        self.b = float(b)
        self.order = int(order)

        self.linear = myLinear1(1, 1)
        self.biases = nn.ModuleList(myBias1(1, 1) for _ in range(order))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _scaled_to_minus_one_one(x, self.a, self.b)

        value = self.linear(x)
        value = self.biases[0](value)
        for bias in self.biases[1:]:
            value = bias(x * value)

        return value


class Horner_IC_1_order(nn.Module):
    """Horner polynomial with hard value condition y(a) = y0."""

    def __init__(self, order: int, a: float, b: float):
        super().__init__()
        if order < 2:
            raise ValueError("Horner_IC_1_order requires order >= 2.")

        self.a = float(a)
        self.b = float(b)
        self.order = int(order)

        self.linear = myLinear1(1, 1)
        self.biases = nn.ModuleList(myBias1(1, 1) for _ in range(order - 1))

    def forward(self, x: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        z = _scaled_to_minus_one_one(x, self.a, self.b)

        value = self.linear(z)
        value = self.biases[0](value)

        zero = _constant_like(0.0, z)
        one = _constant_like(1.0, z)

        # Evaluate the trainable part at z = -1. The correction `left_value`
        # makes the final model exactly satisfy y(a)=y0.
        left_value = self.linear(((-1) ** (self.order + 1)) * one)
        left_value = left_value + ((-1) ** self.order) * self.biases[0](zero)

        for i in range(1, self.order - 1):
            value = self.biases[i](z * value)
            left_value = left_value + ((-1) ** (self.order - i)) * self.biases[i](zero)

        return z * value + y0 + left_value


class Horner_IC_2_order(nn.Module):
    """Horner polynomial with hard conditions y(a)=y0 and y'(a)=y0d."""

    def __init__(self, order: int, a: float, b: float):
        super().__init__()
        if order < 3:
            raise ValueError("Horner_IC_2_order requires order >= 3.")

        self.a = float(a)
        self.b = float(b)
        self.order = int(order)

        self.linear = myLinear1(1, 1)
        self.biases = nn.ModuleList(myBias1(1, 1) for _ in range(order - 2))

    def forward(self, x: torch.Tensor, y0: torch.Tensor, y0d: torch.Tensor) -> torch.Tensor:
        z = _scaled_to_minus_one_one(x, self.a, self.b)

        value = self.linear(z)
        value = self.biases[0](value)
        for i in range(1, self.order - 2):
            value = self.biases[i](z * value)

        zero = _constant_like(0.0, z)
        one = _constant_like(1.0, z)

        # Corrections at z=-1. `left_value` fixes y(a), while `left_derivative`
        # fixes y'(a) after converting the physical derivative to z-coordinates.
        left_value = 0
        left_derivative = 0
        sign_derivative = 1
        sign_value = -1
        derivative_factor = 2

        for i in range(self.order - 3, -1, -1):
            left_derivative = (
                left_derivative
                + derivative_factor * sign_derivative * self.biases[i](zero)
            )
            left_value = left_value + sign_value * self.biases[i](zero)
            sign_derivative *= -1
            sign_value *= -1
            derivative_factor += 1

        left_derivative = left_derivative + sign_derivative * derivative_factor * self.linear(one)
        left_value = left_value + sign_value * self.linear(one)

        # Convert y'(t) to dy/dz because z maps [a,b] to [-1,1].
        a1 = y0d * (self.b - self.a) / 2.0 + left_derivative
        a0 = y0 + a1 + left_value

        value = value * z + a1
        value = value * z + a0
        return value

