# Polynomial Regression for ODEs in Python

This folder contains Python equivalents of the three MATLAB ODE regression
scripts.

The scripts approximate ODE solutions with a polynomial in scaled Taylor form:

```text
P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!
```

The factorial scaling makes derivatives easy to evaluate because each
derivative shifts the coefficient index:

```text
P'(t) = c1 + c2*t/1! + c3*t^2/2! + ...
```

## Associated Paper

This folder contains Python polynomial-regression reference examples used as
supporting material for the associated paper on Horner neural networks. The
final paper reference will be added manually:

```text
TODO: add paper title, authors, venue/year, and link or DOI.
```

## Files

### `ODE_order1_ex1.py`

Python version of `ODE_order1_ex1.m`.

It solves:

```text
x'(t) + 2*x(t) = 1,     x(0) = 1
```

Exact solution:

```text
x(t) = 0.5*(1 + exp(-2*t))
```

### `ODE_order1_ex2.py`

Python version of `ODE_order1_ex2.m`.

It solves:

```text
x'(t) + 2*x(t) = exp(-2*t),     x(0) = 0
```

Exact solution:

```text
x(t) = t*exp(-2*t)
```

### `ODE_order2.py`

Python version of `ODE_order2.m`.

It solves:

```text
x''(t) + 2*x'(t) + x(t) = sin(t),     x(0) = 0, x'(0) = 0
```

Exact solution:

```text
x(t) = (exp(-t) + t*exp(-t) - cos(t))/2
```

### `polynomial_ode_utils.py`

Shared helper functions for:

- building the scaled polynomial basis
- evaluating the polynomial and its derivatives
- computing the RMSE error on the plotting grid
- plotting the model and pointwise error panels
- saving or showing figures

## Main Parameters

Each main script contains:

```python
n = 1 or 2
m = 15
M = 1000 or 10000
```

`n` is the ODE order. For `n = 1`, one initial condition is fixed. For
`n = 2`, two initial conditions are fixed.

`m` is the polynomial degree. With `m = 15`, the model has coefficients
`c0, c1, ..., c15`.

`M` is the number of random collocation points. The ODE is sampled at these
points, giving an overdetermined linear system that is solved with:

```python
ch = np.linalg.pinv(A) @ uk
```

## Requirements

The scripts use:

- NumPy
- Matplotlib

Install them with:

```bash
python3 -m pip install numpy matplotlib
```

## Running

Run the scripts from this folder:

```bash
python3 ODE_order1_ex1.py
python3 ODE_order1_ex2.py
python3 ODE_order2.py
```

Each script prints the polynomial coefficients and the RMSE error, then opens
the plots.

## Useful Options

Use a seed to make the random collocation points repeatable:

```bash
python3 ODE_order1_ex1.py --seed 0
```

Create figures without opening a plot window:

```bash
python3 ODE_order1_ex1.py --seed 0 --no-show
```

Save figures as PNG files:

```bash
python3 ODE_order1_ex1.py --seed 0 --save-dir figures
```

The same options work for all three main scripts.
