# ODE/PDE Poly Comparison

This folder contains small educational examples for solving ordinary
differential equations with neural networks. Each solver script treats the
network as a function approximator:

```text
t  ->  neural network  ->  y(t)
```

PyTorch autograd then computes the derivatives needed in the ODE residual,
for example `y'(t)` or `y''(t)`.

## Files

| File | Role |
| --- | --- |
| `1order_1.py` | Nonlinear first-order ODE: `y * y' = t`, `y(0) = 1`. |
| `1order_2.py` | Linear first-order ODE: `y' + 2y = 1`, `y(0) = 1`. |
| `2order.py` | Linear second-order ODE: `y'' + 2y' + y = sin(t)`, `y(0) = 0`, `y'(0) = 0`. |
| `INR.py` | Model definitions for sigmoid MLP, LeakyReLU MLP, SIREN, and KAN. |
| `My_Fncs.py` | Shared helper functions for autograd derivatives and parameter counting. |
| `model/` | Existing KAN checkpoint/cache output. It is not a source module. |


## Available Models

Each solver script has the same top-level selector:

```python
MODEL_NAME = "kan"  # "leaky_relu", "sigmoid", "siren", or "kan"
```

Supported values:

- `"leaky_relu"` - standard MLP with LeakyReLU hidden activations.
- `"sigmoid"` - standard MLP with sigmoid hidden activations.
- `"siren"` - SIREN network with sine activations.
- `"kan"` - Kolmogorov-Arnold Network from the external `kan` package.

The model construction happens in each script through `build_model(...)`, so
the training loop, ODE residual, plotting, and error reporting stay the same
when you switch architectures.

## Main Configuration

The common training/problem parameters are near the top of each solver script:

```python
SEED = 5
NE = 200
T_START = 0.0
T_END = 3.0
T_INIT = 0.0
EPOCHS = 60
N_PLOT = 5000
```

For `2order.py`, the domain and initial conditions differ:

```python
T_END = 6.0
Y_INIT = 0.0
YD_INIT = 0.0
EPOCHS = 50
```

`NE` is the number of collocation points used for the ODE residual. `N_PLOT`
is only for evaluation and plotting after training.

## Model Parameters

The standard MLP models use:

```python
MLP_WIDTH = 5
MLP_LAYERS = 4
```

The SIREN model uses:

```python
SIREN_WIDTH = 5
SIREN_LAYERS = 3
SIREN_OMEGA = 3.0
```

The KAN model uses:

```python
KAN_WIDTH = [1, 5, 1]
KAN_GRID = 5
KAN_SPLINE_ORDER = 4
KAN_SEED = SEED
KAN_NOISE_SCALE = 0.3
KAN_GRID_RANGE = [T_START, T_END]
KAN_SYMBOLIC_ENABLED = False
KAN_AUTO_SAVE = False
```

Important KAN notes:

- `KAN_WIDTH = [1, 5, 1]` means one input coordinate `t`, one hidden layer of
  width 5, and one output `y(t)`.
- `KAN_GRID_RANGE` should cover the whole ODE domain. In these scripts it is
  tied to `[T_START, T_END]`.
- `KAN_AUTO_SAVE = False` prevents new KAN checkpoint files from being written
  to `./model` on every run.
- Running with `MODEL_NAME = "kan"` requires the external `kan` Python package.

## Loss Function

All examples use a physics-informed loss:

```text
total loss = ODE residual loss + initial-condition loss
```

For the first-order scripts, the initial condition is weighted by:

```python
IC_WEIGHT = 2.0
```

For `2order.py`, value and derivative initial conditions have separate weights:

```python
IC_Y_WEIGHT = 10.0
IC_YD_WEIGHT = 1.0
```

These are soft initial conditions. The model is penalized when the initial
condition is wrong, but the condition is not enforced exactly by the
architecture.

## What Each Script Computes

`1order_1.py` solves:

```text
y * y' = t,    y(0) = 1
exact: y(t) = sqrt(t^2 + 1)
```

`1order_2.py` solves:

```text
y' + 2y = 1,    y(0) = 1
exact: y(t) = 0.5 * (1 + exp(-2t))
```

`2order.py` solves:

```text
y'' + 2y' + y = sin(t),    y(0) = 0,    y'(0) = 0
exact: y(t) = 0.5 * (exp(-t) + t * exp(-t) - cos(t))
```

