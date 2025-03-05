import numpy as np
import bisect
import jax
from jax import grad
import matplotlib.pyplot as plt
import os

jax.config.update("jax_enable_x64", True)

def numerical_gradient(h, x, e=1e-5):
    return (h(x + e) - h(x - e)) / (2 * e)


def automatic_differentiation_JAX(h):
    """
    Use JAX to compute the gradient of function h.
    """
    return jax.jit(grad(h))


def tangents_and_intersections(h, h_prime, x_points):
    """
    Compute the tangents and intersection points for a log-concave function h(x).
    Returns:
    - tangents: A list of tuples (x, h(x), h'(x))
    - intersections: A list of intersection points z_j, where:
        - z_j is the intersection of the tangents at x_points[j] and x_points[j+1].
    """
    tangents = []
    intersections = []
    
    # Calculate tangents and intersections
    for i in range(len(x_points)):
        x = x_points[i]
        h_val = h(x)
        slope = h_prime(x)
        tangents.append((x, h_val, slope))

        # Compute intersection with the next tangent
        if i < len(x_points) - 1:
            x_next = x_points[i + 1]
            h_next = h(x_next)
            slope_next = h_prime(x_next)

            if np.isclose(slope, slope_next, atol=1e-8):
                print(f"Skipping intersection calculation for x={x}, x_next={x_next} due to similar slopes.")
                continue

            # Compute intersection point (guaranteed non-zero denominator)
            z = (h_next - h_val + slope * x - slope_next * x_next) / (slope - slope_next)
            intersections.append(z)

            if i > 0 and z <= intersections[i - 1]:
                raise ValueError(f"Intersections not ordered: z[{i-1}]={intersections[i-1]} >= z[{i}]={z}, which may indicate that h(x) is non-log-concave density over a given domain.")

    return tangents, intersections



def lower_bound(x_points, h, x_star):
    """
    Construct the lower hull using chords 
    and return the lower bound for x_star.

    Parameters:
    - x_points: Sorted list of x-coordinates of the abscissae.
    - h: Function to compute log(f(x)).
    - x_star: Point for which to compute the lower bound.

    Returns:
    - lower_bound: The lower bound value at x_star.
    """
    # Check if x_star is within bounds
    if x_star < x_points[0] or x_star > x_points[-1]:
        return -np.inf

    # Find the interval using binary search
    idx = bisect.bisect_right(x_points, x_star) - 1

    # Compute the slope and intercept for the relevant chord
    x1, x2 = x_points[idx], x_points[idx + 1]
    slope = (h(x2) - h(x1)) / (x2 - x1)
    intercept = h(x1) - slope * x1
    lowerbound = slope * x_star + intercept

    h_val_at_star = h(x_star)
    if h_val_at_star < lowerbound:
        raise ValueError(f"h(x_star)={h_val_at_star} is below lower bound={lowerbound} at x_star={x_star}, which may indicate that h(x) is non-log-concave density over a given domain.")

    return lowerbound



def upper_bound(x_points,h, x_star, tangents, intersections, z0, zk):
    """
    Construct the tangents (upper bound) for h(x).
    Given x_star, return the upper bound at x_star.

    Parameters:
    - h: Function that computes log(f(x)).
    - h_prime: Derivative of h(x).
    - x_points: Sorted list of abscissae (tangent points).
    - x_star: The x-coordinate for which to compute the upper bound.
    - z0, zk : Lower and upper bound of domain of X.

    Returns:
    - upperbound: The value of the upper bound at x_star.
    """

    # Validate input
    if x_star < z0 or x_star > zk:
        raise ValueError(f"x_star={x_star} is out of bounds of domain")

    # Check if x_star is exactly one of the x_points
    if x_star in x_points:
        return h(x_star)

    # Determine which tangent applies
    if x_star <= intersections[0]:
        idx = 0
    elif x_star >= intersections[-1]:
        idx = len(x_points) - 1
    else:
        idx = bisect.bisect_right(intersections, x_star)

    # Compute the upper bound using the tangent at the selected point
    x, h_val, slope = tangents[idx]
    intercept = h_val - slope * x
    upperbound = slope * x_star + intercept

    h_val_at_star = h(x_star)
    if h_val_at_star > upperbound:
        raise ValueError(f"h(x_star)={h_val_at_star} exceeds upper bound={upperbound} at x_star={x_star}, which may indicate that h(x) is non-log-concave density over a given domain.")

    return upperbound


def sample_x_star(tangents, intersections, z0, zk):
    """
    Sample x_star from S_k(x), the normalized piecewise exponential function
    formed by the tangents to h(x).
    """
    segment_weights = []
    total_weight = 0

    # Add z0 and zk to intersections to complete domain
    extended_intersections = [z0] + intersections + [zk]

    for i in range(len(extended_intersections) - 1):
        z1, z2 = extended_intersections[i], extended_intersections[i + 1]
        x, h_val, slope = tangents[i]

        # Validate segment bounds
        if z2 <= z1:
            raise ValueError(f"Invalid segment bounds: z1={z1}, z2={z2}, which may indicate that h(x) is non-log-concave density over a given domain.")

        if np.abs(slope) < 1e-8:  # Handle zero or near-zero slopes
            weight = (z2 - z1) * np.exp(h_val)
        else:
            weight = (1 / slope) * (
                np.exp(h_val + slope * (z2 - x)) - np.exp(h_val + slope * (z1 - x))
            )

        # Validate weight
        if weight < 0 or not np.isfinite(weight):
            raise ValueError(f"Invalid weight computed: weight={weight} for segment [{z1}, {z2}], which may indicate that h(x) is non-log-concave density over a given domain.")

        segment_weights.append(weight)
        total_weight += weight
        
    # Normalize weights
    segment_probs = np.maximum(segment_weights, 0) / total_weight
    
    # Sample a segment index based on probabilities
    segment_idx = np.random.choice(len(segment_probs), p=segment_probs)
    z1, z2 = extended_intersections[segment_idx], extended_intersections[segment_idx + 1]
    
    x, h_val, slope = tangents[segment_idx]

    # Sample within the selected segment
    if np.abs(slope) < 1e-8:
        x_star = np.random.uniform(z1, z2)
    else:
        u = np.random.uniform(0, 1)
        x_star = x + (1 / slope) * np.log(
            np.exp(slope * (z1 - x)) + u * (np.exp(slope * (z2 - x)) - np.exp(slope * (z1 - x)))
        )

    return x_star


def plot_distribution(samples, h, domain, cdf_func, bins=50, output_prefix="plot"):
    # Generate x values for plotting
    x = np.linspace(domain[0], domain[1], 1000)
    target_density = np.exp(h(x))  # Compute target density
    
    # Normalize the target density
    target_density /= np.trapezoid(target_density, x)

    # Ensure the output folder exists
    output_folder = f"plot/{output_prefix}"
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Create a figure for the density comparison
    plt.figure()
    plt.hist(samples, bins=bins, density=True, alpha=0.6, label="Sampled Distribution")
    plt.plot(x, target_density, label="Target Density", color='orange')
    plt.title("Density Comparison")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{output_folder}/{output_prefix}_density.png")  # Save density plot as a file
    plt.close()  # Close the figure to free memory

    # Create a figure for the CDF comparison
    plt.figure()
    empirical_cdf = np.cumsum(np.histogram(samples, bins=100, range=domain, density=True)[0])
    empirical_cdf /= empirical_cdf[-1]
    x_cdf = np.linspace(domain[0], domain[1], len(empirical_cdf))
    plt.plot(x_cdf, empirical_cdf, label="Empirical CDF")
    plt.plot(x, cdf_func(x), label="Target CDF", color='orange')
    plt.title("CDF Comparison")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(f"{output_folder}/{output_prefix}_cdf.png")  # Save CDF plot as a file
    plt.close()  # Close the figure to free memory
