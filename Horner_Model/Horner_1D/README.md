# Horner 1D Hard-IC ODE Examples

This folder contains three small ODE demonstrations using Horner-polynomial
neural-network models. The models are defined in `Model_Horner.py` and the
training scripts use PyTorch autograd to compute derivatives in the ODE
residual.

The key idea is hard initial-condition embedding:

```text
model input:  t
model output: y(t)
initial conditions: built directly into the model forward pass
training loss: ODE residual only
```

## Associated Paper

This folder contains the 1D Horner-network examples that accompany the
associated research paper on Horner neural networks. The final paper reference
will be added manually:

```text
TODO: add paper title, authors, venue/year, and link or DOI.
```

## Files

| File | Purpose |
| --- | --- |
| `Model_Horner.py` | Horner-polynomial model definitions, including first- and second-order hard-IC variants. |
| `My_Fncs.py` | Shared helper functions for autograd derivatives and parameter counting. |
| `our_1order_1.py` | Nonlinear first-order ODE: `y * y' = t`, `y(0) = 1`. |
| `our_1order_2.py` | Linear first-order ODE: `y' + 2y = 1`, `y(0) = 1`. |
| `our_2order.py` | Linear second-order ODE: `y'' + 2y' + y = sin(t)`, `y(0) = 0`, `y'(0) = 0`. |

`__pycache__/` and `.spyproject/` are generated/editor-support folders, not
source code.

## Models

`Model_Horner.py` contains:

- `myBias1` - a small learnable additive coefficient layer.
- `myLinear1` - a small learnable multiplicative coefficient layer.
- `Horner` - a plain Horner-polynomial model without hard initial conditions.
- `Horner_IC_1_order` - Horner model with hard value condition `y(a) = y0`.
- `Horner_IC_2_order` - Horner model with hard value and derivative conditions
  `y(a) = y0`, `y'(a) = y0d`.

All models scale the physical interval `[a, b]` to the polynomial interval
`[-1, 1]`.

## Scripts

`our_1order_1.py` solves:

```text
y * y' = t,    y(0) = 1
exact: y(t) = sqrt(t^2 + 1)
model: Horner_IC_1_order
```

`our_1order_2.py` solves:

```text
y' + 2y = 1,    y(0) = 1
exact: y(t) = 0.5 * (1 + exp(-2t))
model: Horner_IC_1_order
```

`our_2order.py` solves:

```text
y'' + 2y' + y = sin(t),    y(0) = 0,    y'(0) = 0
exact: y(t) = 0.5 * (exp(-t) + t * exp(-t) - cos(t))
model: Horner_IC_2_order
```

## Configuration

Each script has a small configuration block near the top:

```python
SEED = 5
NE = 200
T_START = 0.0
T_END = ...
T_INIT = 0.0
Y_INIT = ...
EPOCHS = 10
N_PLOT = 5000
HORNER_ORDER = ...
```

Meaning:

- `SEED` makes random collocation-point sampling and model initialization
  reproducible.
- `NE` is the number of collocation points used in the ODE residual.
- `T_START`, `T_END` define the ODE domain.
- `T_INIT`, `Y_INIT`, and `YD_INIT` define the initial conditions.
- `EPOCHS` is the number of outer LBFGS steps.
- `N_PLOT` is the dense grid size used for evaluation and plotting.
- `HORNER_ORDER` controls the polynomial order and therefore the number of
  trainable parameters.

## Running

Run any example directly:

```bash
python our_1order_1.py
python our_1order_2.py
python our_2order.py
```

Each script prints:

- the model name,
- the ODE being solved,
- the LBFGS loss after each epoch,
- the number of trainable parameters,
- the RMSE against the exact solution.

The scripts also create Matplotlib figures for the loss, solution, first
derivative, and second derivative.

## Training Pattern

All scripts use the same training structure:

1. Sample random collocation points in the time domain.
2. Build a hard-IC Horner model.
3. Use autograd to compute `y'(t)` and, when needed, `y''(t)`.
4. Compute the ODE residual.
5. Train with LBFGS.
6. Evaluate against the exact solution.

Because the initial conditions are built into the model, the loss does not need
an additional initial-condition penalty term.
