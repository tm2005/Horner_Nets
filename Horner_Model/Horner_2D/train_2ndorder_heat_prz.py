"""Train a 2D Horner model on a one-dimensional heat equation.

Problem:
    y_t - 0.1 y_xx = 0,    x in [0, 1],    t in [0, 3]

Initial condition:
    y(x, 0) = sin(pi x)

Boundary conditions:
    y(0, t) = 0
    y(1, t) = 0

Exact solution:
    y(x, t) = sin(pi x) * exp(-0.1 * pi^2 * t)

The script follows the same layout as the 1D Horner examples:
configuration, PDE definition, dataset, model/optimizer, training, evaluation,
plotting, and a short main block.
"""

import os

import numpy as np
import numpy.matlib
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from functions import count_parameters, partial
from Model_Horner import Horner2d


# ---------------- Configuration ----------------

# Reproducible NumPy sampling for collocation, initial, and boundary points.
SEED = 5

# Number of PDE collocation points.
N_RESIDUAL = 200

# Number of initial-condition points.
N_INITIAL = 50

# Number of boundary-condition points on each boundary.
N_BOUNDARY = 50

# Physical space-time domain.
X_START = 0
X_END = 1
T_START = 0
T_END = 3

# Heat-equation coefficient in y_t - alpha*y_xx = u.
HEAT_ALPHA = 0.1

# Horner2d polynomial order. Larger values give a richer polynomial model.
MODEL_ORDER = 9

# One epoch means one outer LBFGS step. The optimizer can call its closure
# several times inside a single epoch.
EPOCHS = 30

# Evaluation grid size is M_PLOT x M_PLOT.
M_PLOT = 512

# Current loss weights. These preserve the behavior of the existing script.
IC_WEIGHT = 1.0
BC_LEFT_WEIGHT = 1.0
BC_RIGHT_WEIGHT = 1.0


# ---------------- Heat equation ----------------

def rhs(x, t):
    """Right-hand side u(x, t) for y_t - alpha*y_xx = u.

    This example is the homogeneous heat equation, so u = 0.
    """

    # t is part of the standard function signature, but the homogeneous source
    # term does not depend on it.
    del t
    return np.zeros_like(x)


def initial_condition(x):
    """Initial condition y(x, 0) = sin(pi*x)."""

    return np.sin(np.pi * x)


def exact_solution(x, t):
    """Analytical solution used only for evaluation and plotting."""

    return np.sin(np.pi * x) * np.exp(-HEAT_ALPHA * np.pi**2 * t)


def pde_residual(y, yt, yxx, u):
    """Residual of y_t - alpha*y_xx = u."""

    return yt - HEAT_ALPHA * yxx - u


# ---------------- Dataset ----------------

class Dataset1DTransport(Dataset):
    """Dataset that returns all PDE, IC, and BC points as one batch.

    The collocation points `(x, t, u)` are used for the PDE residual.
    The initial-condition points `(xi, gi)` represent `y(x, 0) = g(x)`.
    The boundary time arrays `ti1` and `ti2` are used at `x = 0` and `x = 1`.
    Initial and boundary sets may have different sizes.
    """

    def __init__(self, x, t, u, xi, gi, ti1, ti2):
        self.x = x
        self.t = t
        self.u = u
        self.xi = xi
        self.gi = gi
        self.ti1 = ti1
        self.ti2 = ti2

    def __len__(self):
        # The dataset stores all points in one item because LBFGS is used in a
        # full-batch style.
        return 1

    def __getitem__(self, idx):
        # Compute sizes independently so IC and BC sets do not need to have the
        # same number of points.
        n_residual = np.size(self.u)
        n_initial = np.size(self.gi)
        n_boundary_left = np.size(self.ti1)
        n_boundary_right = np.size(self.ti2)

        # PDE collocation points and their right-hand-side values.
        x = torch.tensor(self.x, dtype=torch.float32).view(n_residual, 1)
        t = torch.tensor(self.t, dtype=torch.float32).view(n_residual, 1)
        u = torch.tensor(self.u, dtype=torch.float32).view(n_residual, 1)

        # Initial condition: xi are spatial points, gi are target values.
        xi = torch.tensor(self.xi, dtype=torch.float32).view(n_initial, 1)
        gi = torch.tensor(self.gi, dtype=torch.float32).view(n_initial, 1)

        # Boundary condition time points for x = 0 and x = 1.
        ti1 = torch.tensor(self.ti1, dtype=torch.float32).view(n_boundary_left, 1)
        ti2 = torch.tensor(self.ti2, dtype=torch.float32).view(n_boundary_right, 1)

        return x, t, u, n_residual, xi, gi, n_initial, ti1, ti2


# Backward-compatible alias for older notebooks or scripts.
Dataset_1d_transp = Dataset1DTransport


# ---------------- Model and optimizer ----------------

def build_model():
    """Create the Horner2d model for the configured space-time domain."""

    return Horner2d(MODEL_ORDER, X_START, X_END, T_START, T_END)


def make_optimizer(model):
    """Create the LBFGS optimizer used by this example."""

    # LBFGS is a second-order optimizer. It is useful here because all training
    # points are available in one batch and the model has a moderate number of
    # parameters.
    return torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=30,
        history_size=60,
        tolerance_grad=1e-12,
        tolerance_change=1e-13,
        line_search_fn="strong_wolfe",
    )


# ---------------- Training data ----------------

def make_training_data():
    """Sample collocation, initial-condition, and boundary-condition data."""

    # Interior collocation points where the PDE residual is minimized.
    t = np.random.uniform(T_START, T_END, N_RESIDUAL)
    x = np.random.uniform(X_START, X_END, N_RESIDUAL)

    u = rhs(x, t)

    # Initial condition points at t = 0.
    xi = np.random.uniform(X_START, X_END, N_INITIAL)
    gi = initial_condition(xi)

    # Boundary condition time points. The left and right boundaries are sampled
    # independently.
    tb1 = np.random.uniform(T_START, T_END, N_BOUNDARY)
    tb2 = np.random.uniform(T_START, T_END, N_BOUNDARY)

    return x, t, u, xi, gi, tb1, tb2


# ---------------- Training ----------------

def train_transport_lbfgs(dataloader, model, optimizer, loss_eq, loss_ic, loss_bc1, loss_bc2, device):
    """Train one outer LBFGS step for the heat-equation PINN.

    The model is trained by minimizing four terms:

    - the PDE residual in the interior,
    - the initial condition at t = 0,
    - the left boundary condition at x = 0,
    - the right boundary condition at x = 1.
    """

    for x, t, u, n_residual, xi, gi, n_initial, ti1, ti2 in dataloader:
        del n_residual, n_initial

        # Residual points need gradients because the PDE uses y_t and y_xx.
        x = x.to(device).clone().detach().requires_grad_(True)
        t = t.to(device).clone().detach().requires_grad_(True)
        u = u.to(device)

        xi = xi.to(device)
        gi = gi.to(device)
        ti1 = ti1.to(device)
        ti2 = ti2.to(device)

        # Fixed coordinates for initial and boundary conditions. The target
        # values for both boundaries are zero in this example.
        t0 = torch.zeros(xi.size()).to(device)
        x0 = torch.zeros(ti1.size()).to(device)
        x1 = torch.ones(ti2.size()).to(device)
        zero_bc_left = torch.zeros(ti1.size()).to(device)
        zero_bc_right = torch.zeros(ti2.size()).to(device)

        def closure():
            # LBFGS may call this closure multiple times during one step.
            optimizer.zero_grad()

            # PDE residual on interior collocation points.
            y = model(x, t)
            yt = partial(y, t)
            yxx = partial(partial(y, x), x)

            # Initial condition: y(x, 0) = sin(pi*x).
            yic = model(xi, t0)

            # Boundary conditions: y(0, t) = 0 and y(1, t) = 0.
            yb1 = model(x0, ti1)
            yb2 = model(x1, ti2)

            # Each MSE term compares one mathematical requirement with its
            # target value.
            loss_pde = loss_eq(pde_residual(y, yt, yxx, u), torch.zeros_like(u))
            loss_i = loss_ic(yic, gi)
            loss_b1 = loss_bc1(yb1, zero_bc_left)
            loss_b2 = loss_bc2(yb2, zero_bc_right)

            # Weights allow the IC and BC terms to be emphasized without
            # changing the PDE residual itself.
            loss = (
                loss_pde
                + IC_WEIGHT * loss_i
                + BC_LEFT_WEIGHT * loss_b1
                + BC_RIGHT_WEIGHT * loss_b2
            )

            loss.backward()
            return loss

        loss = optimizer.step(closure)
        return loss.item()

    raise RuntimeError("Empty dataloader.")


# Backward-compatible alias for the original function name.
train_transport_LBFGS = train_transport_lbfgs


# ---------------- Evaluation and plotting ----------------

def evaluate_solution(model, device):
    """Evaluate the model and exact solution on a dense regular grid."""

    n_plot = M_PLOT * M_PLOT

    # Small offsets avoid evaluating exactly on the domain boundaries in the
    # dense diagnostic grid.
    arrt = np.linspace(T_START + 1e-6, T_END - 1e-6, M_PLOT).reshape(M_PLOT, 1)
    arrx = np.linspace(X_START + 1e-6, X_END - 1e-6, M_PLOT).reshape(M_PLOT, 1)

    # tplot varies along columns, xplot along rows.
    tplot = np.matlib.repmat(arrt.T, M_PLOT, 1).reshape(n_plot, 1)
    xplot = np.matlib.repmat(arrx, 1, M_PLOT).reshape(n_plot, 1)

    tplott = torch.tensor(tplot, dtype=torch.float32).view(n_plot, 1).to(device)
    xplott = torch.tensor(xplot, dtype=torch.float32).view(n_plot, 1).to(device)

    with torch.no_grad():
        y = model(xplott, tplott)
    y = y.to("cpu").detach().numpy()

    yexact = exact_solution(xplot, tplot)

    return arrx, arrt, xplot, tplot, y, yexact


def compute_error_metrics(y, yexact):
    """Compute integrated RMSE, MAE, and maximum absolute error."""

    # The RMSE is scaled by the area element dx*dt, so it behaves like an
    # integrated L2 error over the space-time domain.
    dx = (X_END - X_START) / M_PLOT
    dt = (T_END - T_START) / M_PLOT
    rmse = np.sqrt(np.sum((y - yexact) ** 2) * dx * dt)
    mae = np.mean(np.abs(y - yexact))
    max_error = np.max(np.abs(y - yexact))
    return rmse, mae, max_error


def plot_results(model, device, train_error, arrx, arrt, y, yexact, rmse):
    """Create loss, heatmap, surface, and time-slice plots."""

    # Convert flat evaluation arrays back to grids for imshow and surface plots.
    y_pred_grid = y.reshape(M_PLOT, M_PLOT)
    y_exact_grid = yexact.reshape(M_PLOT, M_PLOT)
    y_error_grid = y_pred_grid - y_exact_grid

    plt.close("all")

    # Training loss history.
    plt.figure(figsize=(8, 4))
    plt.plot(train_error)
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss (LBFGS) - heat equation PINN")
    plt.tight_layout()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = [T_START, T_END, X_START, X_END]

    # Use the same color scale for the predicted and exact heatmaps.
    vmin = min(y_pred_grid.min(), y_exact_grid.min())
    vmax = max(y_pred_grid.max(), y_exact_grid.max())

    im0 = axes[0].imshow(
        y_pred_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("NN solution y(x,t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        y_exact_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Exact sin(pi*x)*exp(-0.1*pi^2*t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    plt.colorbar(im1, ax=axes[1])

    err_abs_max = np.abs(y_error_grid).max()
    im2 = axes[2].imshow(
        y_error_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-err_abs_max,
        vmax=err_abs_max,
    )
    axes[2].set_title(f"Error NN - exact | RMSE = {rmse:.2e}")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()

    # 3D surfaces give another view of the learned and exact solutions.
    fig = plt.figure(figsize=(13, 5))
    t_grid, x_grid = np.meshgrid(arrt.flatten(), arrx.flatten())

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(t_grid, x_grid, y_pred_grid, cmap="viridis", edgecolor="none")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    ax1.set_zlabel("y")
    ax1.set_title("NN solution")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(t_grid, x_grid, y_exact_grid, cmap="viridis", edgecolor="none")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")
    ax2.set_zlabel("y")
    ax2.set_title("Exact solution")
    plt.tight_layout()

    # One-dimensional slices make it easier to compare profiles at fixed times.
    fig, ax = plt.subplots(figsize=(9, 5))
    t_slices = np.linspace(T_START, T_END, 5)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(t_slices)))

    x_line = np.linspace(X_START + 1e-6, X_END - 1e-6, 400).reshape(-1, 1)
    for t_value, color in zip(t_slices, colors):
        t_line = np.full_like(x_line, t_value)
        xt = torch.tensor(x_line, dtype=torch.float32).to(device)
        tt = torch.tensor(t_line, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_line = model(xt, tt).cpu().numpy().flatten()
        y_exact_line = exact_solution(x_line, t_value).flatten()

        ax.plot(x_line, y_exact_line, "-", color=color, lw=2, alpha=0.6)
        ax.plot(x_line, y_line, "--", color=color, lw=1.5, label=f"t = {t_value:.1f}")

    ax.set_xlabel("x")
    ax.set_ylabel("y(x, t)")
    ax.set_title("Time slices: solid = exact, dashed = NN")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()


# ---------------- Main ----------------

if __name__ == "__main__":
    # Use a GPU when available, otherwise keep everything on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    np.random.seed(SEED)

    # Generate all training points before constructing the full-batch dataset.
    x, t, u, xi, gi, tb1, tb2 = make_training_data()

    model = build_model().to(device)
    optimizer = make_optimizer(model)

    # Separate loss objects keep the four mathematical requirements explicit.
    loss_eq = torch.nn.MSELoss()
    loss_ic = torch.nn.MSELoss()
    loss_bc1 = torch.nn.MSELoss()
    loss_bc2 = torch.nn.MSELoss()

    train_data = Dataset1DTransport(x, t, u, xi, gi, tb1, tb2)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

    train_error = [0] * EPOCHS

    print("----------- TRAINING (LBFGS) -----------")
    for epoch in range(EPOCHS):
        train_error[epoch] = train_transport_lbfgs(
            train_dataloader,
            model,
            optimizer,
            loss_eq,
            loss_ic,
            loss_bc1,
            loss_bc2,
            device,
        )

        print(f"Epoch {epoch + 1:4d}  |  Loss = {train_error[epoch]:.3e}")
        # Preserve the current early-stop behavior: stop when the loss is
        # unchanged across consecutive stored values up to a very small
        # tolerance.
        if np.abs(train_error[epoch] - train_error[epoch-1])<1e-16 and np.abs(train_error[epoch-1] - train_error[epoch-2])<1e-16:
            break

    count_parameters(model)

    # Evaluate the trained model against the known analytical solution.
    arrx, arrt, xplot, tplot, y, yexact = evaluate_solution(model, device)
    rmse, mae, max_error = compute_error_metrics(y, yexact)

    print("\n----------- ERROR METRICS -----------")
    print(f"RMSE (L2, integrated):  {rmse:.3e}")
    print(f"MAE  (mean absolute):    {mae:.3e}")
    print(f"max |error|:             {max_error:.3e}")

    plot_results(model, device, train_error, arrx, arrt, y, yexact, rmse)
    plt.show()
