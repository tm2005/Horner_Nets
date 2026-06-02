# Polynomial Regression for ODE Examples

This folder contains three MATLAB/Octave scripts that approximate solutions of
ordinary differential equations (ODEs) using polynomial regression.

The main idea is to write the unknown solution as a polynomial in scaled
Taylor form:

```text
P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!
```

The factorial scaling makes differentiation simple. For example, the
derivative of the polynomial is obtained by shifting the coefficient index:

```text
P'(t) = c1 + c2*t/1! + c3*t^2/2! + ...
```

This is useful because the ODE can be written as a linear system for the
unknown polynomial coefficients.

## Associated Paper

This folder contains MATLAB/Octave polynomial-regression reference examples
used as supporting material for the associated paper on Horner neural networks.
The final paper reference will be added manually:

```text
TODO: add paper title, authors, venue/year, and link or DOI.
```

## Files

### `ODE_order1_ex1.m`

Approximates the first-order ODE:

```text
x'(t) + 2*x(t) = 1,     x(0) = 1
```

The exact solution is:

```text
x(t) = 0.5*(1 + exp(-2*t))
```

The script plots:

- the polynomial approximation and the exact solution
- the pointwise solution error
- the polynomial derivative and the exact derivative
- the pointwise derivative error

It also prints the polynomial coefficients and the RMSE (root mean squared
error) for the solution on the plotting grid.

### `ODE_order1_ex2.m`

Approximates the first-order ODE:

```text
x'(t) + 2*x(t) = exp(-2*t),     x(0) = 0
```

The exact solution is:

```text
x(t) = t*exp(-2*t)
```

The script plots the same types of results as `ODE_order1_ex1.m`.

It also prints the polynomial coefficients and the RMSE (root mean squared
error) for the solution on the plotting grid.

### `ODE_order2.m`

Approximates the second-order ODE:

```text
x''(t) + 2*x'(t) + x(t) = sin(t),     x(0) = 0, x'(0) = 0
```

The exact solution is:

```text
x(t) = (exp(-t) + t*exp(-t) - cos(t))/2
```

The script plots:

- the polynomial approximation and the exact solution
- the pointwise solution error
- the polynomial first derivative and the exact first derivative
- the pointwise first derivative error
- the polynomial second derivative and the exact second derivative
- the pointwise second derivative error

It also prints the polynomial coefficients and the RMSE (root mean squared
error) for the solution on the plotting grid.

## Main Parameters

Each script uses the same three main parameters:

```matlab
n = 1 or 2;
m = 15;
M = 1000 or 10000;
```

### `n`

`n` is the order of the differential equation.

For a first-order ODE, `n = 1`, and one initial condition is fixed:

```text
P(0) = c0
```

For a second-order ODE, `n = 2`, and two initial conditions are fixed:

```text
P(0)  = c0
P'(0) = c1
```

In these scripts, `n` is mainly used as a readable reminder of the ODE order.
The matrix sizes are written explicitly using `m`.

### `m`

`m` is the degree of the polynomial approximation.

For example, when `m = 15`, the polynomial uses the coefficients:

```text
c0, c1, c2, ..., c15
```

A larger `m` gives the polynomial more flexibility, but it can also make the
linear system more sensitive numerically.

### `M`

`M` is the number of random collocation points.

The ODE is sampled at these points, producing an overdetermined linear system:

```text
A*ch is approximately equal to uk
```

The unknown coefficients are computed by:

```matlab
ch = pinv(A)*uk;
```

This gives a least-squares fit of the ODE residual at the sampled points.

## Method Summary

The scripts follow the same workflow:

1. Choose the ODE, initial conditions, polynomial degree `m`, and number of
   collocation points `M`.
2. Generate random collocation points `tk`.
3. Fix the polynomial coefficients required by the initial conditions.
4. Build a matrix `A` that represents the ODE evaluated at the collocation
   points.
5. Solve the linear least-squares problem using `pinv(A)*uk`.
6. Evaluate the polynomial on a dense plotting grid.
7. Compare the polynomial approximation with the exact solution.
8. Print the RMSE for the solution on the plotting grid.
9. Plot the solution, derivative approximations, and pointwise errors.

## Running the Scripts

Open MATLAB/Octave in this folder and run one of the scripts:

```matlab
ODE_order1_ex1
ODE_order1_ex2
ODE_order2
```

Each script creates figures comparing the polynomial model with the exact
solution.

## Reproducibility Note

The scripts use random collocation points through `rand`, so the exact
coefficients and errors may change slightly between runs. To make a run
repeatable, set the random seed before generating `tk`, for example:

```matlab
rng(0)
```
