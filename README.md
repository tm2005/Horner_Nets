# ODE/PDE Poly

This repository contains examples for approximating solutions of 
ordinary differential equations (ODEs) and one partial
differential equation (PDE). The examples compare several approaches:

- neural networks as implicit representations of the function `y(t)`,
- Horner-polynomial models with hard initial-condition embedding,
- a two-dimensional Horner model for the heat equation,
- classical polynomial regression in MATLAB/Octave and Python.

This root README gives a high-level overview of the whole project. More
detailed explanations are available in the `README.md` files inside the
individual subfolders.

## Associated Paper

This repository is intended to accompany a research paper on Horner neural
networks. The final paper reference will be added manually:

```text
TODO: add paper title, authors, venue/year, and link or DOI.
```

If you use this code, please cite the associated paper once the final reference
is available.

## Project Structure

| Folder | Purpose |
| --- | --- |
| `Comparison/` | Comparison of standard INR models for 1D ODE examples. Supported models are sigmoid MLP, LeakyReLU MLP, SIREN, and optional KAN. |
| `Horner_Model/Horner_1D/` | 1D Horner-polynomial models for ODE examples with hard initial conditions. |
| `Horner_Model/Horner_2D/` | 2D Horner model for the heat equation `y_t - 0.1*y_xx = 0`. |
| `Regression_MATLAB/` | MATLAB/Octave polynomial regression for the same ODE test problems. |
| `Regression_Python/` | Python polynomial-regression version of the same ODE test problems. |

Technical folders such as `__pycache__/`, `.spyproject/`, `.git/`, `.codex/`,
and `.agents/` are not part of the numerical source code.

## Main Ideas

### Neural ODE Examples

The `Comparison/` folder treats a neural network as a function approximator:

```text
t -> model -> y(t)
```

The derivatives `y'(t)` and `y''(t)` are computed with PyTorch autograd.
Initial conditions are soft conditions: they are added to the loss function as
penalty terms, but they are not enforced exactly by the model architecture.

Scripts:

- `1order_1.py`: `y*y' = t`, `y(0) = 1`
- `1order_2.py`: `y' + 2y = 1`, `y(0) = 1`
- `2order.py`: `y'' + 2y' + y = sin(t)`, `y(0) = 0`, `y'(0) = 0`

The model is selected near the top of each script through `MODEL_NAME`.
Supported values are `"leaky_relu"`, `"sigmoid"`, `"siren"`, and `"kan"`.

### Horner 1D Models

The `Horner_Model/Horner_1D/` folder solves the same types of ODE problems, but
uses Horner-polynomial models. The key difference is hard initial-condition
embedding:

```text
t -> Horner model -> y(t)
initial conditions are built into the forward pass
the training loss contains only the ODE residual
```

Scripts:

- `our_1order_1.py`: nonlinear ODE `y*y' = t`
- `our_1order_2.py`: linear ODE `y' + 2y = 1`
- `our_2order.py`: linear second-order ODE

The models are defined in `Model_Horner.py`, and helper functions for
derivatives and parameter counting are defined in `My_Fncs.py`.

### Horner 2D Model

The `Horner_Model/Horner_2D/` folder contains an example for the homogeneous
heat equation:

```text
y_t - 0.1*y_xx = 0
x in [0, 1], t in [0, 3]
y(x, 0) = sin(pi*x)
y(0, t) = 0
y(1, t) = 0
```

The exact solution used for comparison is:

```text
y(x, t) = sin(pi*x) * exp(-0.1*pi^2*t)
```

The main script is `train_2ndorder_heat_prz.py`. The `Horner2d` model receives
two inputs, space `x` and time `t`, while the derivatives `y_t` and `y_xx` are
computed with autograd.

### Polynomial Regression

The `Regression_MATLAB/` and `Regression_Python/` folders solve ODE examples
with a polynomial in scaled Taylor form:

```text
P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!
```

Because of the factorial scaling, differentiation becomes an index shift in
the coefficient vector. The coefficients are computed from an overdetermined
linear system using least squares.

## Dependencies

For the Python regression scripts, install:

```bash
python3 -m pip install -r Regression_Python/requirements.txt
```

For the neural-network and Horner PyTorch examples, the required packages are:

- `torch`
- `numpy`
- `matplotlib`
- `pykan`

If `MODEL_NAME = "kan"` is used in `Comparison/`, the additional Python package
`kan` must be installed. For `"leaky_relu"`, `"sigmoid"`, and `"siren"`, this
package is not required. 

The MATLAB scripts can be run in MATLAB or a compatible Octave environment.


## Notes

- `Comparison/model/` contains existing KAN cache/checkpoint output. It is not
  the main source module.
- More detailed descriptions of each equation, configuration, and output are
  available in the README file of the corresponding subfolder.
