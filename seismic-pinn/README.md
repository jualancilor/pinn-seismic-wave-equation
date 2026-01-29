# Physics-Informed Neural Networks for Seismic Wave Equation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A complete implementation of Physics-Informed Neural Networks (PINNs) for solving the seismic wave equation, including both forward and inverse problem capabilities.

## Overview

This project demonstrates:
- **Forward Problem**: Solve wave equation given initial/boundary conditions
- **Inverse Problem**: Recover unknown wave speed from sparse observations
- **Applications**: Seismic wave modeling, subsurface imaging, geophysics

## Physics Background

### Wave Equation

**1D:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

**2D:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

where:
- $u(x,t)$: displacement field
- $c$: wave propagation speed

### PINN Formulation

The neural network $u_\theta(x,t)$ is trained to minimize:

$$\mathcal{L} = \mathcal{L}_{physics} + \lambda_{BC}\mathcal{L}_{BC} + \lambda_{IC}\mathcal{L}_{IC}$$

## Results

### 1D Wave Propagation
<img src="1d_wave.gif" alt="Animasi Gelombang 1D" width="500">

### 2D Wave Propagation
![Animasi Gelombang 2D](2d_wave.gif)

### Inverse Problem: Wave Speed Recovery
][convergence](c_convergence.png)

Successfully recovered wave speed c = 2.0 from initial guess c = 1.0 with < 2% error.

## Project Structure

```
seismic-pinn/
├── notebooks/
│   ├── 01_theory.ipynb           # Mathematical background
│   ├── 02_wave_1d.ipynb          # 1D wave equation
│   ├── 03_wave_2d.ipynb          # 2D wave equation
│   └── 04_inverse_problem.ipynb  # Wave speed recovery
├── results/                       # Generated figures and animations
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/yourusername/seismic-pinn.git
cd seismic-pinn
pip install -r requirements.txt
```

## Quick Start

```bash
# Open notebooks in order:
jupyter notebook notebooks/01_theory.ipynb
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- tqdm
- Jupyter

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.
2. DeepXDE Library: https://deepxde.readthedocs.io/

## License

MIT License
