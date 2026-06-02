# Horner 2D Heat-Equation Example

This folder contains a small educational PyTorch example for training a
two-input Horner-polynomial model on a one-dimensional heat equation. The model
uses space and time as inputs:

```text
(x, t) -> y(x, t)
```

PyTorch autograd is used to compute the derivatives needed in the PDE residual.

## Files

| File | Purpose |
| --- | --- |
| `Model_Horner.py` | Horner-polynomial building blocks and the `Horner2d` model. |
| `functions.py` | Shared autograd helpers and parameter-count reporting. |
| `train_2ndorder_heat_prz.py` | Main training, evaluation, and plotting script. |

Generated folders such as `__pycache__/` and editor folders such as
`.spyproject/` are not part of the source code.

## Mathematical Problem

The script solves the homogeneous heat equation

```text
y_t - 0.1*y_xx = 0
```

on the space-time domain

```text
x in [0, 1],    t in [0, 3]
```

with the initial condition

```text
y(x, 0) = sin(pi*x)
```

and homogeneous Dirichlet boundary conditions

```text
y(0, t) = 0
y(1, t) = 0
```

The exact solution used for error measurement and plots is

```text
y(x, t) = sin(pi*x) * exp(-0.1*pi^2*t)
```

## Model

`Horner2d` is a polynomial model evaluated with Horner's scheme. The outer
polynomial variable is the spatial coordinate `x`. Each coefficient in that
outer polynomial is represented by a one-dimensional Horner polynomial in time
`t`.

The model maps the configured physical intervals to `[-1, 1]` internally:

```python
Horner2d(order, x_start, x_end, t_start, t_end)
```

This keeps the polynomial evaluation numerically better scaled than using the
raw physical coordinates directly.

## Training Loss

The total training loss is the sum of four requirements:

```text
loss = PDE residual + IC loss + left BC loss + right BC loss
```

where

```text
PDE residual = y_t - 0.1*y_xx
```

The derivatives `y_t` and `y_xx` are computed by PyTorch autograd in
`functions.partial`.

The loss weights are configured in `train_2ndorder_heat_prz.py`:

```python
IC_WEIGHT = 1.0
BC_LEFT_WEIGHT = 1.0
BC_RIGHT_WEIGHT = 1.0
```

## Main Configuration

The main settings are near the top of `train_2ndorder_heat_prz.py`:

```python
SEED = 5
N_RESIDUAL = 200
N_INITIAL = 50
N_BOUNDARY = 50
X_START = 0
X_END = 1
T_START = 0
T_END = 3
HEAT_ALPHA = 0.1
MODEL_ORDER = 9
EPOCHS = 30
M_PLOT = 512
```

`N_RESIDUAL` controls the number of interior collocation points used for the
PDE residual. `N_INITIAL` controls the number of initial-condition points at
`t = 0`. `N_BOUNDARY` controls the number of time points on each boundary,
`x = 0` and `x = 1`. The initial and boundary counts are independent and do not
need to be equal.

Increasing `MODEL_ORDER`, `N_RESIDUAL`, or `EPOCHS` can improve accuracy, but it
also increases runtime.

## Running

Run the training script from this folder:

```bash
python train_2ndorder_heat_prz.py
```

The script prints:

- LBFGS loss after each epoch,
- number of trainable parameters,
- integrated RMSE,
- mean absolute error,
- maximum absolute error.

It also creates Matplotlib figures for:

- the loss curve,
- predicted, exact, and error heatmaps,
- predicted and exact 3D surfaces,
- time-slice comparisons.

In a headless terminal environment, Matplotlib may warn that `FigureCanvasAgg`
is non-interactive. That warning only means the figures cannot be shown in that
terminal session; it does not indicate a training error.

## Code Structure

`train_2ndorder_heat_prz.py` is organized in the same style as the earlier 1D
examples:

1. configuration,
2. PDE, initial condition, and exact solution,
3. dataset wrapper,
4. model and optimizer builders,
5. training-data generation,
6. LBFGS training step,
7. evaluation and plotting,
8. main execution block.

The helper aliases `Dataset_1d_transp` and `train_transport_LBFGS` are kept so
older notebooks or scripts can still use the previous names.
