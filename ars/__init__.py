# Import main ARS function
from .ARS import ars

# Import helper functions
from .helper_functions import (
    automatic_differentiation_JAX,
    numerical_gradient,
    tangents_and_intersections,
    sample_x_star,
    lower_bound,
    upper_bound,
    plot_distribution
)

# Define the public API of the package
__all__ = [
    "ars",
    "automatic_differentiation_JAX",
    "numerical_gradient",
    "tangents_and_intersections",
    "sample_x_star",
    "lower_bound",
    "upper_bound",
    "plot_distribution"
]
