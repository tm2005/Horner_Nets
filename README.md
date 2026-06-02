# ODE/PDE Poly

This repository contains small educational examples for approximating
solutions of ordinary differential equations (ODEs) and one partial
differential equation (PDE). The examples compare several approaches:

- neural networks as implicit representations of the function `y(t)`,
- Horner-polynomial models with hard initial-condition embedding,
- a two-dimensional Horner model for the heat equation,
- classical polynomial regression in MATLAB/Octave and Python.

This root README gives a high-level overview of the whole project. More
detailed explanations are available in the `README.md` files inside the
individual subfolders.

For a complete setup tutorial, including Anaconda, Python packages, Spyder,
JupyterLab, and GNU Octave, see [`INSTALLATION.md`](INSTALLATION.md).

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

If `MODEL_NAME = "kan"` is used in `Comparison/`, the additional Python package
`kan` must be installed. For `"leaky_relu"`, `"sigmoid"`, and `"siren"`, this
package is not required.

The MATLAB scripts can be run in MATLAB or a compatible Octave environment.

## Running

Run the INR comparison examples:

```bash
cd Comparison
python 1order_1.py
python 1order_2.py
python 2order.py
```

Run the 1D Horner examples:

```bash
cd Horner_Model/Horner_1D
python our_1order_1.py
python our_1order_2.py
python our_2order.py
```

Run the 2D Horner PDE example:

```bash
cd Horner_Model/Horner_2D
python train_2ndorder_heat_prz.py
```

Run the Python polynomial-regression examples:

```bash
cd Regression_Python
python3 ODE_order1_ex1.py --seed 0
python3 ODE_order1_ex2.py --seed 0
python3 ODE_order2.py --seed 0
```

To check the numerical output without opening plot windows:

```bash
python3 ODE_order1_ex1.py --seed 0 --no-show
```

Run the MATLAB/Octave regression examples:

```matlab
cd Regression_MATLAB
ODE_order1_ex1
ODE_order1_ex2
ODE_order2
```

## Typical Workflow

1. Choose the folder and script that match the equation.
2. Check the configuration block near the top of the script.
3. Adjust the number of collocation points, polynomial order, number of epochs,
   or model type if needed.
4. Run the script from its own folder.
5. Compare the printed loss, RMSE, initial-condition error, and plots.

For fair model comparisons, do not compare only the formal polynomial order or
network width. Also compare the actual number of trainable parameters.

## Notes

- `Comparison/model/` contains existing KAN cache/checkpoint output. It is not
  the main source module.
- The PyTorch scripts set `MPLCONFIGDIR` to `/tmp/matplotlib`, which helps in
  restricted environments where Matplotlib cannot write to the user home
  directory.
- Most scripts display Matplotlib figures. In a terminal without a GUI, a
  warning about non-interactive figures may appear; this is not necessarily a
  numerical error.
- More detailed descriptions of each equation, configuration, and output are
  available in the README file of the corresponding subfolder.
