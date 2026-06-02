"""Neural-network models used by the ODE examples.

INR means "implicit neural representation": the network receives a coordinate,
here time t, and returns the function value y(t). This module contains only
model definitions, so each ODE example can change the problem without copying
the architectures.

The scripts in this folder use four model types:
- a standard MLP with sigmoid activations,
- a standard MLP with LeakyReLU activations,
- SIREN, an MLP with sine activations and special initialization,
- KAN, a Kolmogorov-Arnold Network from the `kan` package.
"""

import torch
import torch.nn as nn

import numpy as np


def _activation_factory(activation):
    """Return an activation class from a simple text name."""

    # Map short configuration names to PyTorch modules. The function returns a
    # class, not an instance, because each layer needs its own activation object.
    activations = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "tanhshrink": nn.Tanhshrink,
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "lrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }

    if isinstance(activation, str):
        key = activation.lower()
        if key not in activations:
            known = ", ".join(sorted(activations))
            raise ValueError(f"Unknown activation '{activation}'. Choose one of: {known}.")
        return activations[key]

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation

    raise TypeError("activation must be a string or torch.nn.Module class")


def _build_mlp(in_feats, mid_feats, no_of_layers, out_feats, activation):
    """Build a simple fully connected MLP.

    Structure:
        input Linear -> activation
        (hidden Linear -> activation) repeated no_of_layers - 1 times
        output Linear

    Here, no_of_layers counts how many Linear layers appear before the output
    layer.
    """

    if no_of_layers < 1:
        raise ValueError("no_of_layers must be at least 1")

    activation_cls = _activation_factory(activation)
    layers = [
        nn.Linear(in_feats, mid_feats),
        activation_cls(),
    ]

    for _ in range(no_of_layers - 1):
        layers.append(nn.Linear(mid_feats, mid_feats))
        layers.append(activation_cls())

    layers.append(nn.Linear(mid_feats, out_feats))
    return nn.Sequential(*layers)


class INR_Sig(nn.Module):
    """MLP that uses sigmoid activations in the hidden layers."""

    def __init__(self, in_feats, mid_feats, no_of_layers, out_feats):
        super(INR_Sig, self).__init__()
        self.net = _build_mlp(in_feats, mid_feats, no_of_layers, out_feats, "sigmoid")
    
    def forward(self, in1):
        """Evaluate the network at the given input coordinates."""

        return self.net(in1)


class INR_LReLU(nn.Module):
    """MLP that uses LeakyReLU activations in the hidden layers."""

    def __init__(self, in_feats, mid_feats, no_of_layers, out_feats):
        super(INR_LReLU, self).__init__()
        self.net = _build_mlp(in_feats, mid_feats, no_of_layers, out_feats, "leaky_relu")

    def forward(self, in1):
        """Evaluate the network at the given input coordinates."""

        return self.net(in1)


class INR_Other_Activations(nn.Module):
    """MLP with the activation selected through the activation argument."""

    def __init__(self, in_feats, mid_feats, no_of_layers, out_feats, activation="leaky_relu"):
        super(INR_Other_Activations, self).__init__()
        self.net = _build_mlp(in_feats, mid_feats, no_of_layers, out_feats, activation)

    def forward(self, in1):
        """Evaluate the network at the given input coordinates."""

        return self.net(in1)


class INR_Other_Acivations(INR_Other_Activations):
    """Backward-compatible name used by the existing scripts."""

    pass


# --------------------------- SIREN MODEL ----------------------------
    
class SineLayer(nn.Module):
    """One SIREN layer: a Linear transformation followed by sin()."""

    # Sitzmann et al. 2020
    # https://arxiv.org/abs/2006.09661
    # 
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    # 
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.
    # 
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """Initialization from the SIREN paper.

        The first layer may use larger weights because it sees the coordinate t
        directly. Hidden layers are scaled by omega_0 so activations and
        gradients stay in a reasonable range.
        """

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        """Apply the linear layer and sine activation."""

        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    """SIREN network for representing the function y(t).

    SIREN is useful when the solution has oscillations or fine details because
    sine activations naturally describe such functions. For smooth ODEs, a
    standard MLP is often enough, but SIREN is useful for comparison.
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30.,
                 hidden_omega_0=30.,
                 rff_mapping_size=None
                 ):
        super().__init__()

        self.net = []
        if rff_mapping_size is None:
            # Standard SIREN: the first layer treats the input as a coordinate.
            self.net.append(SineLayer(in_features, hidden_features,
                                      is_first=True, omega_0=first_omega_0))
        else:
            # This argument is kept for compatibility with older code. This
            # module does not build an explicit RFF map; it only changes the
            # initialization of the first SineLayer.
            self.net.append(SineLayer(in_features, hidden_features,
                                      is_first=False, omega_0=first_omega_0))

        # Add the requested number of hidden sine layers.
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            # A linear output layer often makes real-valued y regression easier.
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        # nn.Sequential keeps forward as a single self.net(coords) call.
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """Return the SIREN approximation at coordinates coords."""

        return self.net(coords)


class INR_KAN(nn.Module):
    """Wrapper around the KAN model from the external `kan` package.

    This wrapper exposes the same PyTorch interface as the other models in this
    module: forward(coords) receives a coordinate tensor with shape (n, 1) and
    returns y(coords). This lets KAN train in the same ODE scripts as the
    sigmoid, LeakyReLU, and SIREN networks.
    """

    def __init__(
        self,
        width,
        grid,
        k,
        seed,
        grid_range,
        noise_scale=0.1,
        symbolic_enabled=False,
        auto_save=False,
        device="cpu",
    ):
        super().__init__()

        try:
            from kan import KAN
        except ImportError as exc:
            raise ImportError(
                "The KAN model requires the `kan` package. "
                "Install it before setting MODEL_NAME = 'kan'."
            ) from exc

        # grid_range must cover the ODE domain. The solver scripts set it to
        # [T_START, T_END] for that reason.
        self.net = KAN(
            width=width,
            grid=grid,
            k=k,
            seed=seed,
            grid_range=grid_range,
            noise_scale=noise_scale,
            symbolic_enabled=symbolic_enabled,
            auto_save=auto_save,
            device=device,
        )

    def forward(self, coords):
        """Evaluate the KAN approximation at the given coordinates."""

        return self.net(coords)


__all__ = [
    "INR_Sig",
    "INR_LReLU",
    "INR_Other_Activations",
    "INR_Other_Acivations",
    "INR_KAN",
    "SineLayer",
    "Siren",
]
