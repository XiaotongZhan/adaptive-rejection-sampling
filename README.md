# Adaptive Rejection Sampling (ARS) Python Package

## Overview
This project implements an **adaptive rejection sampler (ARS)** as described in **Gilks et al. (1992)**. The ARS algorithm is useful for efficiently sampling from log-concave distributions using piecewise linear upper and lower bounds.

This package is designed for ease of use, modularity, and efficiency. It supports numerical differentiation and includes an option for **automatic differentiation (AD)** using **JAX** or **PyTorch**.

## Features
- Accepts **any log-concave density function** as input
- Supports both **numerical differentiation** and **automatic differentiation (AD)**
- Implements **input validation** and **log-concavity checks**
- Provides **vectorized computation** where possible for efficiency
- Includes **unit tests** and **distribution comparison tests**
- Uses Python scientific libraries: `numpy`, `scipy`, `jax`, and `pytorch`
- Provides built-in **plotting tools** for visualizing sampled distributions

## Installation
To install the package, clone this repository and use:
```bash
pip install .
```
Alternatively, you can install dependencies separately using:
```bash
pip install numpy scipy jax torch matplotlib pytest
```

## Usage
A basic example of using the ARS sampler:
```python
import numpy as np
from ars import ars, plot_distribution
from scipy.stats import norm

def log_density(x):
    return -0.5 * x**2  # Standard normal log-density

samples, acceptance_rate = ars(log_density, domain=(-5, 5), n_samples=10000, initial_points=[-2.0, 0.0, 2.0])

print(f"Acceptance Rate: {acceptance_rate:.2%}")
plot_distribution(samples, h=log_density, domain=(-5, 5), cdf_func=norm.cdf, output_prefix="normal")
```

## Testing
To run unit tests:
```bash
pytest test/
```
The package includes tests for:
- **Correctness of sampled distributions** (e.g., Normal, Beta, Gamma, T-Student, Cauchy)
- **Handling of incorrect domains and non-log-concave functions**
- **Functionality of individual components** (e.g., acceptance rate validation, log-density checks)
- **Automatic differentiation (AD) support with JAX and PyTorch**

## References
- Gilks, W. R., & Wild, P. (1992). Adaptive Rejection Sampling for Gibbs Sampling. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*.

## License
This project is licensed under the MIT License.

