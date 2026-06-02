"""Shared helper functions for the Horner 2D examples.

The training script uses these utilities for automatic differentiation and for
reporting the number of trainable model parameters.
"""

import torch


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx with PyTorch autograd.

    The input tensor `x` must have `requires_grad=True`. The returned tensor
    keeps the computation graph so higher derivatives can be computed.

    This is a general helper: it returns the full gradient of `y` with respect to
    `x`. The heat-equation script uses `partial` below because it needs scalar
    partial derivatives such as y_t and y_xx.
    """

    if grad_outputs is None:
        # For scalar-valued outputs with shape `(n, 1)`, ones_like(y) means that
        # every output point contributes equally to the derivative.
        grad_outputs = torch.ones_like(y)

    # create_graph=True is required because the PDE residual needs a second
    # derivative in x.
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True,
    )[0]
    return grad


def partial(y, x):
    """Compute a scalar partial derivative of `y` with respect to `x`.

    The model output is scalar-valued with shape `(n, 1)`, so the derivative
    component we need is the first and only coordinate.

    Example usage in the heat equation:

        yt = partial(y, t)
        yxx = partial(partial(y, x), x)
    """

    component = 0

    # Differentiate the selected output component with respect to the selected
    # input tensor.
    grad = torch.autograd.grad(
        y[..., component],
        x,
        torch.ones_like(y[..., component]),
        create_graph=True,
    )[0]

    # Keep the output shape `(n, 1)` so it matches the model output and loss
    # targets.
    return grad[..., component:component + 1]


def count_parameters(model, verbose=True):
    """Count trainable model parameters.

    By default this also prints the count, matching the reporting style used in
    the ODE comparison scripts.
    """

    # Only parameters with requires_grad=True are optimized and counted.
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print("Number of parameters:")
        print(count)
        print()

    return count
