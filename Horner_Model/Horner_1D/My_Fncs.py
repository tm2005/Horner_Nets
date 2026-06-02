"""Small helper functions shared by all demonstration scripts."""

import torch


def derivative(y, x, grad_outputs=None):
    """Compute dy/dx using PyTorch autograd.

    In the PINN/INR ODE examples, the network gives y(t), and the ODE residual
    needs y'(t) or y''(t). Therefore t must have requires_grad=True; otherwise
    PyTorch does not keep the graph needed for derivatives.
    """

    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True to calculate a derivative.")

    if grad_outputs is None:
        # For one scalar output per point, ones give the ordinary derivative.
        grad_outputs = torch.ones_like(y)

    # create_graph=True is essential: it allows y'' to be computed from y'.
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def count_parameters(model, verbose=True):
    """Count trainable parameters in the model.

    This is useful for model comparisons because width, layer count, and
    architecture type do not always produce the same number of parameters.
    """

    count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not verbose:
        return count

    print("Number of parameters:")
    print(count)
    print("\n")
    return count


def print_all_parameters(model):
    """Print the names and values of all model parameters.

    This is mainly a diagnostic function for small models; for larger networks
    the output can be very long.
    """

    for name, parameter in model.named_parameters():
        print(f"parameter name: {name}")
        print(parameter)
        print("\n")
    return None
