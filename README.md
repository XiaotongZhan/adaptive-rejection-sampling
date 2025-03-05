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

## Installation

To install the package, clone this repository and use:

```bash
pip install .
```

Alternatively, you can install dependencies separately using:

```bash
pip install numpy scipy jax torch
```

## Usage

A basic example of using the ARS sampler:

```python
import numpy as np
from ars import AdaptiveRejectionSampler

def log_density(x):
    return -0.5 * x**2  # Standard normal log-density

sampler = AdaptiveRejectionSampler(log_density)
samples = sampler.sample(n_samples=1000)
print(samples[:10])  # Print first 10 samples
```

## Testing

To run unit tests:

```bash
pytest tests/
```

The package includes tests for:

- **Input validation**
- **Log-concavity checks**
- **Correctness of sampled distributions**
- **Functionality of individual components**

## References

- Gilks, W. R., & Wild, P. (1992). Adaptive Rejection Sampling for Gibbs Sampling. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*.

## License

This project is licensed under the MIT License.

---

This `README.md` provides a professional and structured overview of your project. Let me know if you’d like to customize any section further!

