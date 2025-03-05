import pytest
import numpy as np
from ars import ars, plot_distribution
from scipy.stats import norm, gamma, t, cauchy
from scipy.special import betaln, betainc, gammaln
import jax.numpy as jnp


def test_normal_distribution_ND():
    """
    Test ARS sampling with a Normal Distribution.
    """
    mu = 0
    sigma = 1

    def h_normal(x):
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x - mu)**2 / (2 * sigma**2)

    # Expected properties
    expected_acceptance_rate = 0.9  # Set a reasonable expected acceptance rate threshold

    # Call ARS
    samples, acceptance_rate = ars(h_normal, domain=(-np.inf, np.inf), n_samples=10000, initial_points=[-2.0, 0.0, 2.0])

    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Expected acceptance rate > {expected_acceptance_rate:.2%} but got {acceptance_rate:.2%}"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_normal, domain=(-5, 5), cdf_func=norm.cdf, output_prefix="normal")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")


def test_beta_distribution_ND():
    """
    Test ARS sampling with a Beta Distribution.
    """
    # Parameters for Beta distribution
    alpha, beta = 2.0, 3.0

    def h_beta(x):
        if np.any(x <= 0) or np.any(x >= 1):
            return -np.inf  # Log-density is undefined outside (0, 1)
        return (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x) - betaln(alpha, beta)

    def beta_cdf(x):
        return betainc(alpha, beta, x)

    # Expected properties
    expected_acceptance_rate = 0.9  # Set a reasonable expected acceptance rate threshold

    # Call ARS
    samples, acceptance_rate = ars(h_beta, domain=(1e-5, 1 - 1e-5), n_samples=10000, initial_points=[0.2, 0.5, 0.8])
    
    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Acceptance rate too low: {acceptance_rate:.2%} (Expected > {expected_acceptance_rate:.2%})"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_beta, domain=(1e-5, 1 - 1e-5), cdf_func=beta_cdf, output_prefix="Beta")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")


def test_gamma_distribution_ND():
    """
    Test ARS sampling with a Gamma Distribution.
    """

    # Parameters for Gamma distribution
    shape, scale = 2.0, 2.0  # shape (k), scale (theta)

    def h_gamma(x):
        x = np.asarray(x)
        if np.any(x <= 1e-5):  # Set a strict lower bound to avoid log(0) or near-zero
            return -np.inf
        log_density = (shape - 1) * np.log(x) - x / scale - shape * np.log(scale) - gammaln(shape)
        return log_density

    def gamma_cdf(x):
        return gamma.cdf(x, a=shape, scale=scale)

    # Expected properties
    expected_acceptance_rate = 0.9  # Set a reasonable expected acceptance rate threshold

    # Call ARS
    samples, acceptance_rate = ars(h_gamma, domain=(1e-5, np.inf), n_samples=10000, initial_points=[1.0, 2.0, 5.0])

    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Acceptance rate too low: {acceptance_rate:.2%} (Expected > {expected_acceptance_rate:.2%})"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_gamma, domain=(np.min(samples) * 1.1, np.max(samples) * 1.1), cdf_func=gamma_cdf, output_prefix="Gamma")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")


def test_t_student_distribution_with_correct_domian_ND():
    """
    Test ARS sampling with a T-student Distribution.
    """

    # Degrees of freedom for t-Student distribution
    df = 5

    def h_t_student(x):
        x = np.asarray(x)
        log_density = -0.5 * (df + 1) * np.log(1 + (x**2) / df) - 0.5 * np.log(df * np.pi) - gammaln(0.5 * df) + gammaln(0.5 * (df + 1))
        return log_density

    def t_student_cdf(x):
        return t.cdf(x, df=df)

    # Expected properties
    expected_acceptance_rate = 0.85  # Set a reasonable expected acceptance rate threshold

    # Call ARS
    samples, acceptance_rate = ars(h_t_student, domain=(-2, 2), n_samples=10000, initial_points=[-2.0, 0.0, 2.0])

    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Acceptance rate too low: {acceptance_rate:.2%} (Expected > {expected_acceptance_rate:.2%})"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_t_student, domain=(np.min(samples) * 1.1, np.max(samples) * 1.1), cdf_func=t_student_cdf, output_prefix="T-student")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")

def test_cauchy_distribution_ND():
    """
    Test ARS sampling with a Cauchy Distribution.
    """

    def h_cauchy(x):
        x = np.asarray(x)
        log_density = -np.log(np.pi) - np.log(1 + x**2)  # Log-density of standard Cauchy
        return log_density

    def cauchy_cdf(x):
        return cauchy.cdf(x)  # CDF of the Cauchy distribution

    # Expected properties
    expected_acceptance_rate = 0.85  # Set a reasonable expected acceptance rate threshold

    # Call ARS
    samples, acceptance_rate = ars(h_cauchy, domain=(-1, 1), n_samples=10000, initial_points=np.linspace(-1, 1, 10))

    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Acceptance rate too low: {acceptance_rate:.2%} (Expected > {expected_acceptance_rate:.2%})"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_cauchy, domain=(np.min(samples) * 1.1, np.max(samples) * 1.1), cdf_func=cauchy_cdf, output_prefix="Cauchy")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")


def test_normal_distribution_AD():
    """
    Test ARS sampling with a Normal Distribution using Automatic Differentiation (AD).
    """

    # Parameters for Normal distribution
    mu = 0
    sigma = 1

    def h_normal(x):
        return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - (x - mu)**2 / (2 * sigma**2)

    # Expected properties
    expected_acceptance_rate = 0.85  # Set a reasonable expected acceptance rate threshold

    # Call ARS
    samples, acceptance_rate = ars(h_normal, domain=(-5, 5), n_samples=10000, initial_points=[-2.0, 0.0, 2.0], use_ad=True)

    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Acceptance rate too low: {acceptance_rate:.2%} (Expected > {expected_acceptance_rate:.2%})"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_normal, domain=(-5, 5), cdf_func=norm.cdf, output_prefix="normal_AD")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")

def test_gamma_distribution_AD():
    """
    Test ARS sampling with a Gamma Distribution using Automatic Differentiation (AD).
    """

    shape, scale = 2.0, 2.0  # shape (k), scale (theta)

    def h_gamma(x):
        x = jnp.asarray(x)
        log_density = jnp.where(
            x > 0,
            (shape - 1) * jnp.log(x) - x / scale - shape * jnp.log(scale) - gammaln(shape),
            -jnp.inf  # Log-density undefined for x <= 0
        )
        return log_density

    def gamma_cdf(x):
        from scipy.stats import gamma
        return gamma.cdf(x, a=shape, scale=scale)  # CDF of the Gamma distribution
    
    # Call ARS
    samples, acceptance_rate = ars(h_gamma, domain=(1e-5, np.inf), n_samples=10000, initial_points=[1.0, 2.0, 5.0],use_ad= True)

    # Expected properties
    expected_acceptance_rate = 0.85  # Set a reasonable expected acceptance rate threshold

    # Check acceptance rate
    assert acceptance_rate > expected_acceptance_rate, (
        f"Acceptance rate too low: {acceptance_rate:.2%} (Expected > {expected_acceptance_rate:.2%})"
    )

    # Plot results
    try:
        plot_distribution(samples, h=h_gamma, domain=(np.min(samples) * 1.1, np.max(samples) * 1.1), cdf_func=gamma_cdf, output_prefix="Gamma_AD")
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")

def test_gamma_distribution_wrong_domain_ND():
    """
    Test ARS sampling with a Gamma Distribution using an incorrect domain.
    """

    # Parameters for Gamma distribution
    shape, scale = 2.0, 2.0  # shape (k), scale (theta)

    def h_gamma(x):
        x = np.asarray(x)
        if np.any(x <= 0):  # Gamma function is not defined for x <= 0
            return -np.inf
        log_density = (shape - 1) * np.log(x) - x / scale - shape * np.log(scale) - gammaln(shape)
        return log_density

    # Call ARS with an incorrect domain (including negative values where Gamma is not defined)
    with pytest.raises(ValueError) as exc_info:
        samples, acceptance_rate = ars(h_gamma, domain=(-10, 10), n_samples=1000, initial_points=[-10.0, 2.0, 5.0])

    # Check if the correct error is raised
    assert "h(x) is non-log-concave density over a given domain" in str(exc_info.value), (
        "ARS did not raise an error with the correct message for an incorrect domain."
    )

def test_t_student_distribution_wrong_domain_ND():
    """
    Test ARS sampling with a t_student Distribution using an incorrect domain.
    """

    df = 5

    def h_t_student(x):
        x = np.asarray(x)
        log_density = -0.5 * (df + 1) * np.log(1 + (x**2) / df) - 0.5 * np.log(df * np.pi) - gammaln(0.5 * df) + gammaln(0.5 * (df + 1))
        return log_density

    with pytest.raises(ValueError) as exc_info:
        samples, acceptance_rate = ars(h_t_student, domain=(-5, 5), n_samples=1000, initial_points=[-2.0, 0.0, 2.0])

    # Check if the correct error is raised
    assert "h(x) is non-log-concave density over a given domain" in str(exc_info.value), "The error message does not match the expected format."




def test_cauchy_distribution_wrong_domain_ND():
    """
    Test ARS sampling with a Log-Cauchy Distribution using an incorrect domain.
    The domain includes non-positive values where the Log-Cauchy distribution is undefined.
    """
    # Incorrect domain that includes non-positive values
    def h_cauchy(x):
        x = np.asarray(x)
        log_density = -np.log(np.pi) - np.log(1 + x**2)  # Log-density of standard Cauchy
        return log_density

    domain = (-10, 10)
    initial_points = [-2, 0, 2]

    # Expect a ValueError when calling ARS with an incorrect domain
    with pytest.raises(ValueError) as exc_info:
        samples, acceptance_rate = ars(h_cauchy, domain=domain, n_samples=1000, initial_points=initial_points)

    # Check if the raised error contains the correct message
    assert "h(x) is non-log-concave density over a given domain" in str(exc_info.value), (
        "ARS did not raise an error with the correct message for an incorrect domain."
    )

def test_quadratic_distribution_ND():
    """
    Test ARS sampling with a Quadratic Distribution.
    The quadratic distribution is not log-concave and should fail.
    """
    def quadratic(x):
        return x**2

    domain = (-10, 10)
    initial_points = [-2, 0, 2]

    # Expect a ValueError when calling ARS with a non log-concave distribution
    with pytest.raises(ValueError) as exc_info:
        samples, acceptance_rate = ars(quadratic, domain=domain, n_samples=1000, initial_points=initial_points)

    # Check if the correct error is raised
    assert "h(x) is non-log-concave density over a given domain" in str(exc_info.value), (
        "ARS did not raise an error with the correct message for a non log-concave distribution."
    )

def test_sinusoidal_distribution_ND():
    """
    Test ARS sampling with a Sinusoidal Distribution.
    The sinusoidal distribution is not log-concave and should fail.
    """
    def sinfunc(x):
        return np.sin(x)

    domain = (-10, 10)
    initial_points = [-2, 0, 2]

    # Expect a ValueError when calling ARS with a non log-concave distribution
    with pytest.raises(ValueError) as exc_info:
        samples, acceptance_rate = ars(sinfunc, domain=domain, n_samples=1000, initial_points=initial_points)

    # Check if the correct error is raised
    assert "h(x) is non-log-concave density over a given domain" in str(exc_info.value), (
        "ARS did not raise an error with the correct message for a non log-concave distribution."
    )