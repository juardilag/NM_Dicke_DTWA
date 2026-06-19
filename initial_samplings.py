"""
Initial-condition samplers for the Discrete Truncated Wigner Approximation (DTWA).

The DTWA approximates the initial quantum state by an ensemble of classical phase-
space points drawn from (a discrete approximation of) the Wigner distribution.
This module provides the two samplers used by the Dicke-model solver:

* :func:`discrete_spin_sampling_factorized` -- discrete Wigner sampling of the
  collective spin built from N independent two-level atoms.
* :func:`cavity_wigner_sampling` -- Gaussian Wigner sampling of the cavity coherent
  field amplitude.

Both take a JAX PRNGKey so that each trajectory in a vmapped ensemble samples an
independent initial condition.
"""

import jax.numpy as jnp
import jax


def discrete_spin_sampling_factorized(key: jax.Array, initial_direction: jax.Array,
                                      n_spins: int = 1) -> jax.Array:
    """Sample a collective-spin initial vector via factorized discrete Wigner.

    Each of the ``n_spins`` atoms is initialized as a spin-1/2 coherent state
    pointing along ``initial_direction``. In the discrete Wigner scheme the two
    transverse components of each atom independently take the discrete values
    +/- 1 with probability 1/2 (so each transverse quadrature has unit variance,
    reproducing the spin-1/2 Wigner fluctuations). Summing over the N atoms gives
    the collective spin, with mean N along the chosen direction and transverse
    fluctuations that grow as sqrt(N) (central-limit behavior).

    The returned vector uses the Pauli (length-N) scaling; callers that want the
    collective spin J typically divide by 2.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey for this sample (split internally for the two transverse axes).
    initial_direction : jax.Array, shape (3,)
        Mean spin direction (need not be normalized; it is normalized internally).
    n_spins : int, optional
        Number of two-level atoms N. Default 1.

    Returns
    -------
    s_init : jax.Array, shape (3,)
        Sampled collective-spin vector in Pauli scaling:
        ``N * mean_dir + f1 * perp1 + f2 * perp2``, where ``f1, f2`` are sums of
        ``n_spins`` independent +/- 1 draws along two orthonormal axes
        perpendicular to ``mean_dir``.
    """
    k1, k2 = jax.random.split(key)

    # 1. Directional normalization
    initial_direction = jnp.array(initial_direction, dtype=float)
    mean_vec = initial_direction / (jnp.linalg.norm(initial_direction) + 1e-12)

    # 2. Perpendicular axes
    v = jnp.where(jnp.abs(mean_vec[0]) < 0.9,
                  jnp.array([1.0, 0.0, 0.0]),
                  jnp.array([0.0, 1.0, 0.0]))

    perp1 = jnp.cross(mean_vec, v)
    perp1 = perp1 / (jnp.linalg.norm(perp1) + 1e-12)
    perp2 = jnp.cross(mean_vec, perp1)

    # 3. Discrete fluctuations (+/- 1)
    flips1 = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=(n_spins,)) - 1.0
    flips2 = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=(n_spins,)) - 1.0

    f1 = jnp.sum(flips1)
    f2 = jnp.sum(flips2)

    # Returns vector of length N (Pauli scaling)
    s_init = (n_spins * mean_vec) + (f1 * perp1) + (f2 * perp2)

    return s_init


def cavity_wigner_sampling(key: jax.Array, alpha_initial: complex) -> jax.Array:
    """Sample the initial complex cavity field amplitude from the Wigner distribution.

    Adds vacuum quantum fluctuations to the mean coherent amplitude. The Wigner
    function of a coherent state is a Gaussian of variance 1/2 in each quadrature
    (real and imaginary), so independent Gaussian noise with standard deviation
    sqrt(1/2) is added to each.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey for this sample (split internally for the two quadratures).
    alpha_initial : complex
        Mean initial cavity amplitude (the coherent-state displacement).

    Returns
    -------
    alpha_0 : jax.Array, complex128 scalar
        ``alpha_initial + N(0, 1/2) + i N(0, 1/2)``.
    """
    k1, k2 = jax.random.split(key)
    # Variance 0.5 per quadrature means std_dev = sqrt(0.5)
    fluc_re = jax.random.normal(k1) * jnp.sqrt(0.5)
    fluc_im = jax.random.normal(k2) * jnp.sqrt(0.5)

    # [FIX] Cast directly to complex128. DO NOT take the square root.
    alpha_0 = jnp.array(alpha_initial, dtype=jnp.complex128)

    return alpha_0 + fluc_re + 1j * fluc_im


def discrete_cavity_sampling(key: jax.Array, alpha_initial: complex) -> jax.Array:
    """Discrete-Wigner sampling of the initial cavity amplitude.

    Like :func:`cavity_wigner_sampling` but each quadrature is drawn from the
    two-point distribution +/- sqrt(1/2) instead of a Gaussian. This reproduces
    the coherent-state Wigner mean (0) and variance (1/2 per quadrature) exactly
    while suppressing the large-deviation tails, which can reduce the estimator
    variance of two-point correlators -- the same rationale as discrete TWA for
    spin-1/2 (Schachenmayer et al.). Higher moments differ from the true Gaussian,
    so it is an approximation that trades tail fidelity for lower sampling noise.

    Parameters
    ----------
    key : jax.Array
        JAX PRNGKey (split internally for the two quadratures).
    alpha_initial : complex
        Mean initial cavity amplitude.

    Returns
    -------
    alpha_0 : jax.Array, complex128 scalar
        ``alpha_initial + (+/- sqrt(1/2)) + i (+/- sqrt(1/2))``.
    """
    k1, k2 = jax.random.split(key)
    amp = jnp.sqrt(0.5)
    fluc_re = (2.0 * jax.random.bernoulli(k1, p=0.5) - 1.0) * amp
    fluc_im = (2.0 * jax.random.bernoulli(k2, p=0.5) - 1.0) * amp
    alpha_0 = jnp.array(alpha_initial, dtype=jnp.complex128)
    return alpha_0 + fluc_re + 1j * fluc_im
