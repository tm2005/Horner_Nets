"""Hard-IC Horner example for the nonlinear first-order ODE.

Problem:
    y(t) * y'(t) = t,    y(0) = 1

Exact solution:
    y(t) = sqrt(t^2 + 1)

This script mirrors the structure of `1order_1.py`, but uses
`Horner_IC_1_order` from `Model_Horner.py`. The initial condition is embedded
directly in the model, so the loss contains only the ODE residual.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# Use a writable Matplotlib config directory in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from Model_Horner import Horner_IC_1_order
from My_Fncs import count_parameters, derivative


# ---------------- Configuration ----------------

SEED = 5

# Number of collocation points used in the ODE residual.
NE = 200

# Physical time domain and initial condition.
T_START = 0.0
T_END = 3.0
T_INIT = 0.0
Y_INIT = 1.0

# One epoch means one outer LBFGS step. Each step can evaluate the closure
# multiple times internally.
EPOCHS = 20

# Dense grid used only for plotting and RMSE evaluation after training.
N_PLOT = 5000

# Polynomial order for the hard-IC Horner model.
HORNER_ORDER = 12


# ---------------- Differential equation ----------------

def rhs(t):
    return t


def ode_residual(y, yd, u):
    return y * yd - u


def exact_solution(t):
    return np.sqrt(t**2 + 1.0)


def exact_first_derivative(t):
    return t / np.sqrt(t**2 + 1.0)


def exact_second_derivative(t):
    return 1.0 / (np.sqrt(t**2 + 1.0) ** 3)


# ---------------- Dataset ----------------

class Dataset1stHardIC(Dataset):
    """Dataset that returns all collocation points as a single batch."""

    def __init__(self, t, u, t_init, y_init):
        self.t = t
        self.u = u
        self.t_init = t_init
        self.y_init = y_init

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        n = np.size(self.u)

        # Shape (n, 1) matches a scalar-input neural representation y(t).
        t = torch.tensor(self.t, dtype=torch.float32).view(n, 1)
        u = torch.tensor(self.u, dtype=torch.float32).view(n, 1)

        # Keep the initial condition in the dataset so the training loop can
        # pass it directly into the hard-IC model.
        t_init = torch.tensor(self.t_init, dtype=torch.float32).view(1, 1)
        y_init = torch.tensor(self.y_init, dtype=torch.float32).view(1, 1)

        return t, u, t_init, y_init


# ---------------- Model and optimizer ----------------

def build_model():
    """Create the hard-IC Horner model for this ODE domain."""

    return Horner_IC_1_order(HORNER_ORDER, T_START, T_END)


def make_optimizer(model):
    """Create the LBFGS optimizer used by this demonstration."""

    return torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=30,
        history_size=90,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )


# ---------------- Training ----------------

def train_1st_order_hard_ic_lbfgs(dataloader, model, optimizer, loss_eq, device):
    """Train one outer LBFGS step for the first-order hard-IC model."""

    for t, u, t_init, y_init in dataloader:
        # The model embeds the initial condition value but does not need t_init.
        del t_init

        # t must require gradients because the ODE residual uses y'(t).
        t = t.to(device).clone().detach().requires_grad_(True)
        u = u.to(device)
        y_init = y_init.to(device)

        def closure():
            # LBFGS calls the closure repeatedly during line search.
            optimizer.zero_grad()

            # Hard-IC model call: y_init is an input to the model, not a loss term.
            y = model(t, y_init)
            yd = derivative(y, t)

            residual = ode_residual(y, yd, u)
            loss = loss_eq(residual, torch.zeros_like(residual))

            loss.backward()
            return loss

        loss = optimizer.step(closure)
        return loss.item()

    raise RuntimeError("Empty dataloader.")


# ---------------- Evaluation and plotting ----------------

def evaluate_solution(model, device):
    """Evaluate y, y', and y'' on a dense grid for plots and RMSE."""

    tplot = np.linspace(T_START, T_END, N_PLOT, endpoint=True).reshape(N_PLOT, 1)
    tplott = torch.tensor(tplot, dtype=torch.float32, requires_grad=True).view(N_PLOT, 1)
    tplott = tplott.to(device)

    y_init = torch.tensor(Y_INIT, dtype=torch.float32, device=device).view(1, 1)

    y = model(tplott, y_init)
    yd = derivative(y, tplott)
    ydd = derivative(yd, tplott)

    return (
        tplot,
        y.to("cpu").detach().numpy(),
        yd.to("cpu").detach().numpy(),
        ydd.to("cpu").detach().numpy(),
    )


def plot_results(train_error, tplot, y, yd, ydd):
    """Create diagnostic plots and return the integral RMSE."""

    yexact = exact_solution(tplot)
    yexactd = exact_first_derivative(tplot)
    yexactdd = exact_second_derivative(tplot)

    plt.close("all")

    plt.figure()
    plt.plot(train_error)
    plt.yscale("log")
    plt.grid()
    plt.title("Loss (LBFGS)")

    plt.figure()
    plt.plot(tplot, yexact, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, y, "b", label="Horner hard-IC")
    plt.grid()
    plt.legend()
    plt.title("y and exact solution")

    plt.figure()
    plt.plot(tplot, yexactd, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, yd, "b", label="Horner hard-IC")
    plt.grid()
    plt.legend()
    plt.title("dy/dt")

    plt.figure()
    plt.plot(tplot, yexactdd, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, ydd, "b", label="Horner hard-IC")
    plt.grid()
    plt.legend()
    plt.title("d^2y/dt^2")

    dt = (T_END - T_START) / N_PLOT
    rmse = np.sqrt(np.sum((y - yexact) ** 2) * dt)
    return rmse


# ---------------- Main ----------------

if __name__ == "__main__":
    # Use GPU when available, otherwise run on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Random collocation points are sorted only for easier diagnostics.
    t = np.sort(np.random.uniform(T_START, T_END, NE))
    u = rhs(t)

    model = build_model().to(device)
    optimizer = make_optimizer(model)
    loss_eq = torch.nn.MSELoss()

    train_data = Dataset1stHardIC(t, u, T_INIT, Y_INIT)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=False)

    train_error = [0.0] * EPOCHS

    print("Model: Horner_IC_1_order")
    print("ODE: y * y' = t, y(0) = 1")

    for epoch in range(EPOCHS):
        train_error[epoch] = train_1st_order_hard_ic_lbfgs(
            train_dataloader, model, optimizer, loss_eq, device
        )
        print(f"Epoch {epoch + 1}\n Loss : {train_error[epoch]:.3e}\n-----------------------------")
        if np.abs(train_error[epoch] - train_error[epoch-1])<1e-16 and np.abs(train_error[epoch-1] - train_error[epoch-2])<1e-16:
            break

    count_parameters(model)

    tplot, y, yd, ydd = evaluate_solution(model, device)
    rmse = plot_results(train_error, tplot, y, yd, ydd)
    print(f"RMSE: {rmse}\n")
