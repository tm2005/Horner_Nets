"""Horner-polynomial models for the 2D space-time heat-equation example.

The main model in this folder is `Horner2d`, a two-input polynomial model for
functions of space and time:

    (x, t) -> y(x, t)

The polynomial is evaluated with Horner's scheme. In `Horner2d`, the outer
Horner recurrence is in the spatial variable `x`, while the coefficients are
small one-dimensional Horner models in time `t`.

The class names `myBias1` and `myLinear1` are kept unchanged so older scripts or
notebooks that import them still work.
"""

import torch
import torch.nn as nn


class myBias1(nn.Module):
    """Learnable additive coefficient used inside the Horner recurrence.

    The layer computes

        output = input + b * out_scale

    where `b` is trainable. The parameter is initialized uniformly around
    `rand_dist_mean` with half-width `rand_dist_range`.

    The `in_features` and `out_features` arguments make this helper look like a
    standard PyTorch layer, although the examples in this folder use scalar
    inputs and outputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_scale=1.0,
        rand_dist_range=0.01,
        rand_dist_mean=0,
    ):
        super(myBias1, self).__init__()

        # Trainable coefficient b. The formula below samples from
        # [rand_dist_mean - rand_dist_range, rand_dist_mean + rand_dist_range].
        self.b = nn.Parameter(
            2 * rand_dist_range * torch.rand(
                out_features,
                in_features,
                dtype=torch.float32,
            ).view(out_features, in_features)
            - rand_dist_range
            + rand_dist_mean,
            requires_grad=True,
        )
        self.out_scale = out_scale

    def forward(self, x):
        """Add the scaled trainable coefficient to the current polynomial value."""

        return x + self.b * self.out_scale


class myLinear1(nn.Module):
    """Learnable multiplicative coefficient used as a Horner leading term.

    The layer computes

        output = matmul(input, a) * out_scale

    In these examples the input and output dimensions are both one, so this is
    effectively a trainable scalar multiplication.

    This is the multiplicative counterpart of `myBias1`; together they provide
    the trainable coefficients used by the Horner recurrence.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_scale=1.0,
        rand_dist_range=0.01,
        rand_dist_mean=0,
    ):
        super(myLinear1, self).__init__()

        # Trainable coefficient a. The initialization range matches `myBias1`.
        self.a = nn.Parameter(
            2 * rand_dist_range * torch.rand(
                out_features,
                in_features,
                dtype=torch.float32,
            ).view(out_features, in_features)
            - rand_dist_range
            + rand_dist_mean,
            requires_grad=True,
        )
        self.out_scale = out_scale

    def forward(self, x):
        """Apply the scaled trainable multiplication to `x`."""

        return torch.matmul(x, self.a) * self.out_scale


class Horner(nn.Module):
    """One-dimensional polynomial evaluated by Horner's scheme.

    Arguments:
        order: Polynomial order.
        a, b: Input interval. The forward pass maps this interval to [-1, 1].

    This model is used directly as the time-dependent coefficient model inside
    `Horner2d`.

    Conceptually, it represents a polynomial in the scaled input variable z:

        c_0 + z*(c_1 + z*(c_2 + ...))

    but it evaluates the expression in nested Horner form for a compact and
    stable implementation.
    """

    def __init__(self, order, a, b):
        super(Horner, self).__init__()

        self.a = a
        self.b = b
        self.order = order

        # Leading coefficient term in the one-dimensional polynomial.
        self.linear = myLinear1(1, 1)

        # Remaining additive coefficient terms in the nested Horner expression.
        self.biases = nn.ModuleList()
        for i in range(order):
            self.biases.append(myBias1(1, 1))

    def forward(self, x):
        """Evaluate the one-dimensional Horner polynomial at points `x`."""

        # Scale the input interval [a, b] to [-1, 1].
        x = 2 / (self.b - self.a) * x - (self.b + self.a) / (self.b - self.a)

        # Horner recurrence:
        # a[n]*x -> a[n-1] + a[n]*x -> a[n-2] + x*(...) -> ...
        x1 = self.linear(x)
        x1 = self.biases[0](x1)
        for i in range(1, self.order):
            x1 = self.biases[i](x * x1)

        return x1


class Horner2d(nn.Module):
    """Two-dimensional Horner model for y(x, t).

    The spatial direction is the outer Horner variable. Each spatial
    coefficient is itself a one-dimensional Horner polynomial in time.

    Arguments:
        order: Outer polynomial order in `x`.
        ax, bx: Spatial interval.
        at, bt: Time interval.

    The resulting structure is a polynomial in the scaled spatial coordinate,
    with time-dependent coefficients:

        p(x, t) = h_0(t) + x*(h_1(t) + x*(h_2(t) + ...))
    """

    def __init__(self, order, ax, bx, at, bt):
        super(Horner2d, self).__init__()

        self.ax = ax
        self.bx = bx
        self.at = at
        self.bt = bt
        self.order = order

        # Leading x-dependent term of the outer spatial polynomial.
        self.linear = myLinear1(1, 1)

        # Time-dependent coefficient functions h_i(t) for the outer x polynomial.
        self.Horners = nn.ModuleList()
        for i in range(order):
            self.Horners.append(Horner(i + 1, at, bt))

    def forward(self, x, t):
        """Evaluate the two-dimensional Horner model at coordinates `(x, t)`."""

        # Scale both physical coordinates to [-1, 1] before polynomial evaluation.
        x = 2 / (self.bx - self.ax) * x - (self.bx + self.ax) / (self.bx - self.ax)
        t = 2 / (self.bt - self.at) * t - (self.bt + self.at) / (self.bt - self.at)

        # Outer Horner recurrence in x, where every coefficient is a Horner
        # polynomial in t.
        p = self.Horners[0](t) + self.linear(x)
        for i in range(1, self.order):
            p = self.Horners[i](t) + x * p

        return p
