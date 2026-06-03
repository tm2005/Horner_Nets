"""Horner-spline example for the linear second-order ODE.

Problem:
    y''(t) + 2 y'(t) + y(t) = sin(t),    y(0) = 0, y'(0) = 0

Exact solution:
    y(t) = 0.5 * (exp(-t) + t * exp(-t) - cos(t))
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from Model_Horner_Spline import HornerSplineIC2
from My_Fncs import (
    count_parameters,
    derivative,
    evaluate_piecewise_solution,
    rmse_error,
    select_subdomain,
    should_stop,
)


# ---------------- Configuration ----------------

SEED = 5
NE = 200

T_START = 0.0
T_END = 6.0
T_INIT = 0.0
Y_INIT = 0.0
YD_INIT = 0.0

EPOCHS = 10
N_PLOT = 5000

BREAKPOINTS = [T_START, 2.0, T_END]
HORNER_ORDERS = [10, 10]


# ---------------- Differential equation ----------------

def rhs(t):
    return np.sin(t)


def ode_residual(y, yd, ydd, u):
    return ydd + 2.0 * yd + y - u


def exact_solution(t):
    return 0.5 * (np.exp(-t) + t * np.exp(-t) - np.cos(t))


def exact_first_derivative(t):
    return 0.5 * (np.sin(t) - t * np.exp(-t))


def exact_second_derivative(t):
    return 0.5 * (t * np.exp(-t) - np.exp(-t) + np.cos(t))


# ---------------- Dataset ----------------

class Dataset2ndSplineHardIC(Dataset):
    """Dataset that returns all collocation points as a single batch."""

    def __init__(self, t, u, t_init, y_init, yd_init):
        self.t = t
        self.u = u
        self.t_init = t_init
        self.y_init = y_init
        self.yd_init = yd_init

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        n = np.size(self.u)
        t = torch.tensor(self.t, dtype=torch.float32).view(n, 1)
        u = torch.tensor(self.u, dtype=torch.float32).view(n, 1)
        t_init = torch.tensor(self.t_init, dtype=torch.float32).view(1, 1)
        y_init = torch.tensor(self.y_init, dtype=torch.float32).view(1, 1)
        yd_init = torch.tensor(self.yd_init, dtype=torch.float32).view(1, 1)
        return t, u, t_init, y_init, yd_init


# ---------------- Model and optimizer ----------------

def build_model():
    return HornerSplineIC2(BREAKPOINTS, HORNER_ORDERS, initial_conditions=(Y_INIT, YD_INIT))


def make_optimizer(model):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=40,
        history_size=90,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )


# ---------------- Training ----------------

def residual_loss_2nd_order(t, u, model, interval_idx, loss_eq):
    left, right = model.intervals[interval_idx]
    is_last = interval_idx == model.n_intervals - 1
    t_sub, u_sub = select_subdomain(t, u, left, right, is_last)

    y = model.evaluate_interval(interval_idx, t_sub)
    yd = derivative(y, t_sub)
    ydd = derivative(yd, t_sub)
    residual = ode_residual(y, yd, ydd, u_sub)
    return loss_eq(residual, torch.zeros_like(residual))


def train_2nd_order_spline_lbfgs(dataloader, model, optimizer, loss_eq, device):
    for t, u, t_init, y_init, yd_init in dataloader:
        del t_init, y_init, yd_init

        t = t.to(device).clone().detach().requires_grad_(True)
        u = u.to(device)

        def closure():
            optimizer.zero_grad()
            losses = [
                residual_loss_2nd_order(t, u, model, interval_idx, loss_eq)
                for interval_idx in range(model.n_intervals)
            ]
            total_loss = torch.stack(losses).sum()
            total_loss.backward()
            closure.last_sub_losses = [loss.item() for loss in losses]
            return total_loss

        loss = optimizer.step(closure)
        return loss.item(), closure.last_sub_losses

    raise RuntimeError("Empty dataloader.")


# ---------------- Evaluation and plotting ----------------

def evaluate_solution(model, device):
    tplot = np.linspace(T_START, T_END, N_PLOT, endpoint=True).reshape(N_PLOT, 1)
    tplott = torch.tensor(tplot, dtype=torch.float32, requires_grad=True).view(N_PLOT, 1)
    tplott = tplott.to(device)

    y, yd, ydd = evaluate_piecewise_solution(tplott, model)
    return (
        tplot,
        y.to("cpu").detach().numpy(),
        yd.to("cpu").detach().numpy(),
        ydd.to("cpu").detach().numpy(),
    )


def plot_results(train_error, tplot, y, yd, ydd):
    yexact = exact_solution(tplot)
    yexactd = exact_first_derivative(tplot)
    yexactdd = exact_second_derivative(tplot)
    boundaries = BREAKPOINTS[1:-1]

    plt.close("all")

    plt.figure()
    plt.plot(train_error)
    plt.yscale("log")
    plt.grid()
    plt.title("Loss (Horner spline LBFGS)")

    plt.figure()
    plt.plot(tplot, yexact, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, y, "b", label="Horner spline hard-IC")
    for boundary in boundaries:
        plt.axvline(x=boundary, color="gray", linestyle=":", alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title("y and exact solution")

    plt.figure()
    plt.plot(tplot, yexactd, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, yd, "b", label="Horner spline hard-IC")
    for boundary in boundaries:
        plt.axvline(x=boundary, color="gray", linestyle=":", alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title("dy/dt")

    plt.figure()
    plt.plot(tplot, yexactdd, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, ydd, "b", label="Horner spline hard-IC")
    for boundary in boundaries:
        plt.axvline(x=boundary, color="gray", linestyle=":", alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title("d^2y/dt^2")

    return rmse_error(y, yexact)


# ---------------- Main ----------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t = np.sort(np.random.uniform(T_START, T_END, NE))
    u = rhs(t)

    model = build_model().to(device)
    optimizer = make_optimizer(model)
    loss_eq = torch.nn.MSELoss()

    train_data = Dataset2ndSplineHardIC(t, u, T_INIT, Y_INIT, YD_INIT)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=False)
    train_error = [0.0] * EPOCHS

    print("Model: HornerSplineIC2")
    print("ODE: y'' + 2y' + y = sin(t), y(0) = 0, y'(0) = 0")
    print(f"Breakpoints: {BREAKPOINTS}")

    for epoch in range(EPOCHS):
        train_error[epoch], sub_losses = train_2nd_order_spline_lbfgs(
            train_dataloader, model, optimizer, loss_eq, device
        )
        sub_loss_text = ", ".join(f"L{idx + 1}: {loss:.3e}" for idx, loss in enumerate(sub_losses))
        print(
            f"Epoch {epoch + 1}\n"
            f" Loss : {train_error[epoch]:.3e}\n"
            f" {sub_loss_text}\n"
            "-----------------------------"
        )
        if should_stop(train_error, epoch):
            break

    count_parameters(model)
    tplot, y, yd, ydd = evaluate_solution(model, device)
    rmse = plot_results(train_error, tplot, y, yd, ydd)
    print(f"RMSE: {rmse}\n")
    plt.show()

