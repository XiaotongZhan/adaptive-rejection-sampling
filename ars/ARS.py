import numpy as np
import bisect

from .helper_functions import *


def ars(h, h_prime=None, domain=(1e-5, 1), n_samples=1000, initial_points=None, use_ad= False):

    """
    Adaptive Rejection Sampling (ARS) algorithm for efficiently sampling from a log-concave probability density function.

    Parameters:
    -----------
    h : callable
        The log of the target density function (log(f(x))).
        When using AD, make sure to define function using jnp instead of np.
    h_prime : callable, optional
        The derivative of the log-density function, h'(x). If not provided, numerical differentiation or automatic differentiation (with JAX) will be used.
    domain : tuple, optional
        The lower and upper bounds of the domain for the sampling process. Default is (1e-5, 1).
    n_samples : int, optional
        The number of samples to generate. Default is 1000.
    initial_points : list, optional
        A list of initial abscissae (x-coordinates) to construct the initial hulls. Default is [-2.0, 0.0, 2.0].
    use_ad : bool, optional
        Whether to use automatic differentiation (with JAX) for computing h'(x). Default is False.

    Returns:
    --------
    samples : list
        A list of sampled points drawn from the target distribution.
    acceptance_rate : float
        The proportion of proposed samples that were accepted during the sampling process.

    Notes:
    ------
    - This algorithm assumes that the target log-density function h(x) is concave. If h(x) is not concave, the method will not work correctly.
    - Make sure to use jnp instead np when defining h(x) if decide to use_ad=TRUE.
    """

    if initial_points is None:
        initial_points = [-2.0, 0.0, 2.0]

    x_points = sorted(initial_points)
    samples = []
    
    # determine bounds of domain
    z0, zk = domain[0], domain[1]

    if h_prime is None:
        if use_ad:
            h_prime = automatic_differentiation_JAX(h)  # Use automatic differentiation
            print("Note: Using AD usually is longer, one may consider reduce the number of iterations...")
        else:
            h_prime = lambda x: numerical_gradient(h, x)  # Use numerical differentiation

    # Initialize counters
    total_candidates = 0  # Total number of candidates generated
    accepted_samples = 0  # Total number of accepted samples

    while len(samples) < n_samples:
        # Construct tangents and intersections (zjs)
        tangents, intersections = tangents_and_intersections(h, h_prime, x_points)

        # Sample a candidate
        x_star = sample_x_star(tangents, intersections, z0,zk)
        total_candidates += 1

        # Squeezing test
        lowerbound = lower_bound(x_points, h, x_star)
        upperbound = upper_bound(x_points, h, x_star, tangents, intersections, z0,zk)

        
        if np.random.uniform(0, 1) < np.exp(lowerbound - upperbound):
            samples.append(x_star)
            accepted_samples += 1  # Increment accepted sample count
            continue

        # Rejection test
        if np.random.uniform(0, 1) < np.exp(h(x_star) - upperbound):
            samples.append(x_star)
            accepted_samples += 1  # Increment accepted sample count

        # Update abscissae
        if x_star not in x_points:
            bisect.insort(x_points, x_star)
    
    acceptance_rate = accepted_samples / total_candidates

    return samples, acceptance_rate
