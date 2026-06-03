# Horner 1D Spline Hard-IC ODE Examples

This folder contains piecewise Horner-polynomial versions of the three
`Horner_1D` examples.

The original `Horner_1D` scripts use one hard-IC Horner polynomial on the full
domain. These scripts split the domain into intervals and place one hard-IC
Horner polynomial on each interval.

## Files

| File | Purpose |
| --- | --- |
| `Model_Horner.py` | Base Horner models and hard initial-condition variants. |
| `Model_Horner_Spline.py` | Piecewise spline wrappers `HornerSplineIC1` and `HornerSplineIC2`. |
| `My_Fncs.py` | Autograd derivatives, parameter counting, RMSE, and spline evaluation helpers. |
| `our_1order_1.py` | Nonlinear first-order ODE: `y * y' = t`, `y(0) = 1`. |
| `our_1order_2.py` | Linear first-order ODE: `y' + 2y = 1`, `y(0) = 1`. |
| `our_2order.py` | Linear second-order ODE: `y'' + 2y' + y = sin(t)`, `y(0) = 0`, `y'(0) = 0`. |

## Spline Paradigm

The domain is split into intervals:

```text
[a, g1], [g1, g2], ..., [gk, b]
```

Each interval has its own Horner polynomial. The left boundary conditions for
each interval are propagated from the previous interval:

- `HornerSplineIC1` propagates the value `y`, giving hard `C0` continuity.
- `HornerSplineIC2` propagates both `y` and `y'`, giving hard `C1` continuity.

The ODE residual is still trained with LBFGS, but the residual loss is computed
separately on each interval and then summed.

## Running

Run from this folder:

```bash
python our_1order_1.py
python our_1order_2.py
python our_2order.py
```

Each script prints:

- the model name,
- the ODE,
- spline breakpoints,
- total LBFGS loss,
- per-interval losses,
- trainable parameter count,
- RMSE against the exact solution.

The plots include vertical gray lines at the internal spline breakpoints.

