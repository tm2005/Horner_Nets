"""Educational example: linear second-order ODE.

Problem:
    y''(t) + 2 y'(t) + y(t) = sin(t),    y(0) = 0, y'(0) = 0

This example extends the first two scripts to a second-order problem. The
network still learns y(t), but autograd now computes both y'(t) and y''(t).
The initial conditions are soft terms in the loss function: one term penalizes
y(0), and the other penalizes y'(0).
"""

# Torch: model, optimization, and Dataset/DataLoader come from PyTorch.
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# Standard libraries for numerical work and plotting.
import os
import numpy as np

# Matplotlib sometimes tries to write configuration under the home directory.
# /tmp is a safe location in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

# Shared helper functions: autograd derivative and parameter count.
from My_Fncs import count_parameters, derivative

# Neural-network architectures that can be compared.
from INR import INR_KAN, INR_LReLU, INR_Sig, Siren


# ---------------- Configuration ----------------

# Change only MODEL_NAME for a quick architecture comparison.
MODEL_NAME = "kan"  # "leaky_relu", "sigmoid", "siren", or "kan"

# SEED makes collocation-point sampling and model initialization reproducible.
SEED = 5

# NE is the number of points where the ODE residual is penalized.
NE = 200

# Solution domain and initial conditions.
T_START = 0.0
T_END = 6.0
T_INIT = 0.0
Y_INIT = 0.0
YD_INIT = 0.0

# Here, one LBFGS epoch means one outer optimizer.step call.
EPOCHS = 50

# Dense point grid used only for plotting and post-training error measurement.
N_PLOT = 5000

# Standard MLP settings.
MLP_WIDTH = 5
MLP_LAYERS = 4

# SIREN model settings.
SIREN_WIDTH = 5
SIREN_LAYERS = 3
SIREN_OMEGA = 3.0

# KAN model settings.
# width=[1, 5, 1] means: one input t, one hidden layer of width 5, one output y.
KAN_WIDTH = [1, 5, 1]

# grid is the number of spline-grid intervals, and k is the spline order used by KAN.
KAN_GRID = 5
KAN_SPLINE_ORDER = 4

# KAN has its own seed. Keep SEED if you want the same seed as the rest of the script.
KAN_SEED = SEED
KAN_NOISE_SCALE = 0.3

# The spline grid must cover the full domain where the ODE is trained and evaluated.
KAN_GRID_RANGE = [T_START, T_END]

# The symbolic part is not needed for these numerical demonstrations, and
# auto_save=False prevents KAN from writing checkpoint files to ./model on each run.
KAN_SYMBOLIC_ENABLED = False
KAN_AUTO_SAVE = False

# Weights of the soft initial conditions in the total loss.
IC_Y_WEIGHT = 10.0
IC_YD_WEIGHT = 1.0


# ---------------- Differential equation ----------------

# ODE: y'' + 2y' + y = sin(t), y(0) = 0, y'(0) = 0
# Exact solution: y(t) = 0.5 * (exp(-t) + t*exp(-t) - cos(t))
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

class Dataset2ndInit(Dataset):
    """Dataset that returns all collocation points as one sample.

    For the second-order problem, we store both y(0) and y'(0), because both
    conditions enter the loss function.
    """

    def __init__(self, t, u, t_init, y_init, yd_init):
        self.t = t
        self.u = u
        self.t_init = t_init
        self.y_init = y_init
        self.yd_init = yd_init

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # DataLoader asks for an index, but this dataset has only one sample.
        n = np.size(self.u)

        # Shape (n, 1) matches a network that receives one coordinate t per row.
        t = torch.tensor(self.t, dtype=torch.float32).view(n, 1)
        u = torch.tensor(self.u, dtype=torch.float32).view(n, 1)

        # Initial conditions are specified at the single point t_init.
        t_init = torch.tensor(self.t_init, dtype=torch.float32).view(1, 1)
        y_init = torch.tensor(self.y_init, dtype=torch.float32).view(1, 1)
        yd_init = torch.tensor(self.yd_init, dtype=torch.float32).view(1, 1)

        return t, u, t_init, y_init, yd_init


# ---------------- Models ----------------

def build_model(model_name, device="cpu"):
    """Build a model from the text name in the configuration."""

    key = model_name.lower()

    if key in ("leaky_relu", "leakyrelu", "lrelu"):
        return INR_LReLU(
            in_feats=1,
            mid_feats=MLP_WIDTH,
            no_of_layers=MLP_LAYERS,
            out_feats=1,
        )

    if key in ("sigmoid", "sigmoide"):
        return INR_Sig(
            in_feats=1,
            mid_feats=MLP_WIDTH,
            no_of_layers=MLP_LAYERS,
            out_feats=1,
        )

    if key == "siren":
        return Siren(
            in_features=1,
            hidden_features=SIREN_WIDTH,
            hidden_layers=SIREN_LAYERS,
            out_features=1,
            outermost_linear=True,
            first_omega_0=SIREN_OMEGA,
            hidden_omega_0=SIREN_OMEGA,
            rff_mapping_size=None,
        )

    if key == "kan":
        return INR_KAN(
            width=KAN_WIDTH,
            grid=KAN_GRID,
            k=KAN_SPLINE_ORDER,
            seed=KAN_SEED,
            grid_range=KAN_GRID_RANGE,
            noise_scale=KAN_NOISE_SCALE,
            symbolic_enabled=KAN_SYMBOLIC_ENABLED,
            auto_save=KAN_AUTO_SAVE,
            device=str(device),
        )

    raise ValueError("MODEL_NAME must be 'leaky_relu', 'sigmoid', 'siren', or 'kan'.")


# ---------------- Training ----------------

def make_optimizer(model):
    """Configure the LBFGS optimizer.

    LBFGS is often useful for small PINN/ODE demonstrations because it uses a
    second-order approximation. The closure can be called multiple times during
    one step, so it should have no side effects except gradients.
    """

    return torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=30,
        history_size=90,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )


def train_2nd_order_ic_loss_lbfgs(dataloader, model, optimizer, loss_eq, loss_ic, device):
    """Train one outer LBFGS step for a second-order ODE."""

    for t, u, t_init, y_init, yd_init in dataloader:
        # t must be a leaf tensor with requires_grad=True so autograd can give y'(t) and y''(t).
        t = t.to(device).clone().detach().requires_grad_(True)
        u = u.to(device)

        # t_init also needs gradients because the initial condition contains y'(t_init).
        t_init = t_init.to(device).clone().detach().requires_grad_(True)
        y_init = y_init.to(device)
        yd_init = yd_init.to(device)

        def closure():
            # LBFGS can call the closure multiple times, so gradients are cleared here.
            optimizer.zero_grad()

            # The neural network represents y(t); autograd builds y'(t) and y''(t).
            y = model(t)
            yd = derivative(y, t)
            ydd = derivative(yd, t)

            # Physics loss: the residual should be zero at all collocation points.
            residual = ode_residual(y, yd, ydd, u)
            loss = loss_eq(residual, torch.zeros_like(residual))

            # Soft initial conditions: penalize both value and first derivative at t=0.
            y_init_model = model(t_init)
            yd_init_model = derivative(y_init_model, t_init)
            loss = loss + IC_Y_WEIGHT * loss_ic(y_init_model, y_init)
            loss = loss + IC_YD_WEIGHT * loss_ic(yd_init_model, yd_init)

            loss.backward()
            return loss

        loss = optimizer.step(closure)
        return loss.item()

    raise RuntimeError("Empty dataloader.")


# ---------------- Evaluation and plotting ----------------

def evaluate_solution(model, device):
    """Evaluate y, y', and y'' on a dense point grid for plots."""

    tplot = np.linspace(T_START, T_END, N_PLOT, endpoint=True).reshape(N_PLOT, 1)

    # requires_grad=True remains enabled because evaluation also needs derivatives.
    tplott = torch.tensor(tplot, dtype=torch.float32, requires_grad=True).view(N_PLOT, 1)
    tplott = tplott.to(device)

    y = model(tplott)
    yd = derivative(y, tplott)
    ydd = derivative(yd, tplott)

    return (
        tplot,
        y.to("cpu").detach().numpy(),
        yd.to("cpu").detach().numpy(),
        ydd.to("cpu").detach().numpy(),
    )


def plot_results(train_error, tplot, y, yd, ydd):
    """Plot loss, solution, and derivatives, then return an integral RMSE measure."""

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
    plt.plot(tplot, y, "b", label="NN")
    plt.grid()
    plt.legend()
    plt.title("y and exact solution")

    plt.figure()
    plt.plot(tplot, yexactd, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, yd, "b", label="NN")
    plt.grid()
    plt.legend()
    plt.title("dy/dt")

    plt.figure()
    plt.plot(tplot, yexactdd, "-.r", linewidth=3, label="exact")
    plt.plot(tplot, ydd, "b", label="NN")
    plt.grid()
    plt.legend()
    plt.title("d^2y/dt^2")

    # Discrete integral error; dt turns the pointwise sum into an integral approximation.
    dt = (T_END - T_START) / N_PLOT
    rmse = np.sqrt(np.sum((y - yexact) ** 2) * dt)
    return rmse


# ---------------- Main ----------------

if __name__ == "__main__":
    # Use a GPU if one is available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Collocation points are random, but sorted for easier plotting and diagnostics.
    t = np.sort(np.random.uniform(T_START, T_END, NE))
    u = rhs(t)

    model = build_model(MODEL_NAME, device).to(device)
    optimizer = make_optimizer(model)

    loss_eq = torch.nn.MSELoss()
    loss_ic = torch.nn.MSELoss()

    train_data = Dataset2ndInit(t, u, T_INIT, Y_INIT, YD_INIT)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=False)

    train_error = [0.0] * EPOCHS

    print(f"Model: {MODEL_NAME}")
    print("ODE: y'' + 2y' + y = sin(t), y(0) = 0, y'(0) = 0")

    # Each epoch is one LBFGS step over the full point set.
    for epoch in range(EPOCHS):
        train_error[epoch] = train_2nd_order_ic_loss_lbfgs(
            train_dataloader, model, optimizer, loss_eq, loss_ic, device
        )
        print(f"Epoch {epoch + 1}\n Loss : {train_error[epoch]:.3e}\n-----------------------------")
        if np.abs(train_error[epoch] - train_error[epoch-1])<1e-16 and np.abs(train_error[epoch-1] - train_error[epoch-2])<1e-16:
            break
        
    count_parameters(model)

    tplot, y, yd, ydd = evaluate_solution(model, device)
    rmse = plot_results(train_error, tplot, y, yd, ydd)
    print(f"RMSE: {rmse}\n")

    torch.set_printoptions(precision=7)
    # Final check of both initial conditions after training.
    t_init = torch.tensor([[T_INIT]], dtype=torch.float32, device=device, requires_grad=True)
    y_init_model = model(t_init)
    yd_init_model = derivative(y_init_model, t_init)
    print(f"Value at t = {T_INIT} is {y_init_model.item():.7f}.")
    print(f"Derivative at t = {T_INIT} is {yd_init_model.item():.7f}.")
