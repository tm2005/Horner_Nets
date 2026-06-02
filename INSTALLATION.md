# Installation Guide

This guide explains how to set up the software needed to run the examples in
this repository:

- Python examples in `Comparison/`, `Horner_Model/`, and `Regression_Python/`
- MATLAB/Octave scripts in `Regression_MATLAB/`
- optional IDE tools such as Spyder and JupyterLab

The recommended Python setup uses Anaconda Distribution and a dedicated conda
environment. Keeping the project in its own environment avoids dependency
conflicts with other Python projects.

## 1. Install Anaconda Distribution

Download Anaconda Distribution from the official Anaconda installation page:

```text
https://www.anaconda.com/docs/getting-started/anaconda/install
```

Choose the installer for your operating system.

### Windows

1. Download the Windows graphical installer.
2. Run the installer.
3. Choose "Just Me" unless you specifically need a system-wide installation.
4. After installation, open **Anaconda Prompt** from the Start menu.
5. Check that conda works:

```bash
conda --version
```

### macOS

1. Download the macOS graphical `.pkg` installer.
2. Run the installer.
3. Open Terminal.
4. Check that conda works:

```bash
conda --version
```

### Linux

1. Download the Linux shell installer from the official Anaconda page.
2. Open a terminal in the folder where the installer was downloaded.
3. Run the installer. The exact file name can change between Anaconda
   releases, so use the name of the file you downloaded:

```bash
bash Anaconda3-*-Linux-x86_64.sh
```

4. Close and reopen the terminal, or reload the shell configuration:

```bash
source ~/.bashrc
```

5. Check that conda works:

```bash
conda --version
```

## 2. Create a Project Environment

Open Anaconda Prompt on Windows, or a normal terminal on macOS/Linux.

Create a new environment named `ode-pde-poly`:

```bash
conda create -n ode-pde-poly python=3.11 pip -y
```

Activate it:

```bash
conda activate ode-pde-poly
```

After activation, the prompt should show the environment name:

```text
(ode-pde-poly)
```

Use this environment whenever you run the scripts in this repository.

## 3. Install Core Python Packages

Install the basic scientific packages:

```bash
conda install numpy matplotlib -y
```

These packages are enough for:

- `Regression_Python/`
- plotting utilities used throughout the Python examples

Install PyTorch for the neural-network and Horner examples.

For a CPU-only setup on Windows or Linux, use:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For macOS, use:

```bash
python -m pip install torch torchvision torchaudio
```

For an NVIDIA GPU/CUDA setup, do not guess the CUDA command manually. Use the
official PyTorch selector:

```text
https://pytorch.org/get-started/locally/
```

This project is small enough to run on CPU, so GPU support is optional.

## 4. Optional: Install KAN Support

The `Comparison/` scripts support four model types:

```text
leaky_relu, sigmoid, siren, kan
```

The first three use only PyTorch. The `kan` option requires the external
PyKAN package because `Comparison/INR.py` imports:

```python
from kan import KAN
```

Install PyKAN only if you want to run scripts with:

```python
MODEL_NAME = "kan"
```

Recommended command:

```bash
python -m pip install pykan
```

Alternative installation directly from GitHub:

```bash
python -m pip install git+https://github.com/KindXiaoming/pykan.git
```

If KAN installation creates dependency conflicts, the simplest workaround is
to use one of the non-KAN models by setting `MODEL_NAME` to `"sigmoid"`,
`"leaky_relu"`, or `"siren"` at the top of the script.

## 5. Optional: Install Spyder IDE

Spyder is a convenient IDE for this project because it works well with
scientific Python, plots, and interactive variables.

Install Spyder inside the project environment:

```bash
conda activate ode-pde-poly
conda install spyder -y
```

Launch it:

```bash
spyder
```

When running a script in Spyder, set the working directory to the folder that
contains the script. This matters because several scripts use local imports
such as:

```python
from My_Fncs import derivative
from Model_Horner import Horner_IC_1_order
```

Examples:

- for `Comparison/1order_1.py`, set the working directory to `Comparison/`
- for `Horner_Model/Horner_1D/our_1order_1.py`, set it to
  `Horner_Model/Horner_1D/`
- for `Horner_Model/Horner_2D/train_2ndorder_heat_prz.py`, set it to
  `Horner_Model/Horner_2D/`
- for `Regression_Python/ODE_order1_ex1.py`, set it to `Regression_Python/`

You can also launch Spyder from Anaconda Navigator. If Spyder is installed in
more than one environment, make sure it uses the `ode-pde-poly` interpreter.

## 6. Optional: Install JupyterLab

JupyterLab is useful if you want to create notebooks around the scripts.

Install it:

```bash
conda activate ode-pde-poly
conda install jupyterlab ipykernel -y
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name ode-pde-poly --display-name "Python (ode-pde-poly)"
```

Launch JupyterLab:

```bash
jupyter lab
```

## 7. Verify the Python Installation

From the root of this repository, activate the environment:

```bash
conda activate ode-pde-poly
cd /path/to/ODE_PDE/Poly
```

Check the core packages:

```bash
python -c "import numpy, matplotlib, torch; print('Python packages OK'); print('torch:', torch.__version__)"
```

If you installed KAN, check it too:

```bash
python -c "from kan import KAN; print('KAN package OK')"
```

Run a quick Python regression example without opening plot windows:

```bash
cd Regression_Python
python ODE_order1_ex1.py --seed 0 --no-show
```

Run a PyTorch example from its own folder:

```bash
cd ../Horner_Model/Horner_1D
python our_1order_2.py
```

Most PyTorch scripts create Matplotlib figures. If you run them from a terminal
without a graphical display, Matplotlib may warn that figures cannot be shown
interactively. That warning does not necessarily mean that the numerical run
failed.

## 8. Install GNU Octave for MATLAB Scripts

The files in `Regression_MATLAB/` are MATLAB/Octave scripts. If you do not
have MATLAB, install GNU Octave.

Official Octave download page:

```text
https://octave.org/download.html
```

### Windows

1. Go to the official Octave download page.
2. Download the Windows installer.
3. Run the installer.
4. Open **GNU Octave** from the Start menu.
5. Check the installation in the Octave command window:

```matlab
version
```

### macOS

The Octave project points macOS users to the Octave wiki and third-party
package managers such as Homebrew or MacPorts.

With Homebrew:

```bash
brew install octave
```

Then check:

```bash
octave --version
```

### Ubuntu/Debian Linux

```bash
sudo apt update
sudo apt install octave
```

Then check:

```bash
octave --version
```

### Fedora Linux

```bash
sudo dnf install octave
```

Then check:

```bash
octave --version
```

### Arch Linux

```bash
sudo pacman -S octave
```

Then check:

```bash
octave --version
```

## 9. Run the MATLAB/Octave Scripts

Open Octave and change into the MATLAB regression folder:

```matlab
cd /path/to/ODE_PDE/Poly/Regression_MATLAB
```

Run one script by typing its name without `.m`:

```matlab
ODE_order1_ex1
ODE_order1_ex2
ODE_order2
```

You can also run a script from a terminal:

```bash
cd /path/to/ODE_PDE/Poly/Regression_MATLAB
octave ODE_order1_ex1.m
```

The scripts print polynomial coefficients and RMSE values, then create figures
for the approximation, exact solution, derivatives, and pointwise errors.

## 10. Quick Command Summary

Minimal Python setup, CPU-only on Windows/Linux:

```bash
conda create -n ode-pde-poly python=3.11 pip -y
conda activate ode-pde-poly
conda install numpy matplotlib -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Minimal Python setup on macOS:

```bash
conda create -n ode-pde-poly python=3.11 pip -y
conda activate ode-pde-poly
conda install numpy matplotlib -y
python -m pip install torch torchvision torchaudio
```

Add Spyder:

```bash
conda install spyder -y
spyder
```

Add JupyterLab:

```bash
conda install jupyterlab ipykernel -y
jupyter lab
```

Add optional KAN support:

```bash
python -m pip install pykan
```

Install Octave on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install octave
```

## Official References

- Anaconda installation:
  `https://www.anaconda.com/docs/getting-started/anaconda/install`
- Conda environments:
  `https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html`
- PyTorch installation selector:
  `https://pytorch.org/get-started/locally/`
- Spyder with Anaconda:
  `https://www.anaconda.com/docs/getting-started/guides/ides/spyder`
- GNU Octave downloads:
  `https://octave.org/download.html`
- PyKAN:
  `https://github.com/KindXiaoming/pykan`
