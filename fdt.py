"""
Two-time correlations, linear-response functions, and spectra for the explicit-
cavity DTWA solver, used to test the fluctuation-dissipation theorem (FDT).

Given the trajectory solver in :mod:`dtwa_non_integrated`, this module measures,
in the (post-transient) stationary window:

* Connected correlations C(tau) = <delta x(t+tau) delta x(t)> for the spin
  observable S_x and the cavity observable Re(psi_c).
* Linear-response (susceptibility) functions chi(tau) via the impulsive-kick
  method: propagate a base ensemble and a perturbed ensemble that share the same
  RNG keys, and take the normalized difference of the means.
* Their Fourier transforms, the noise power spectrum S(omega) and the dissipative
  response Im chi(omega), whose ratio probes the FDT.

Symmetry folding
----------------
Means and correlations use *folded* trajectories (each trajectory mapped into the
positive Z2 well) so the broken-symmetry baseline does not cancel. Responses use
*raw* (un-folded) trajectories, because the impulsive perturbation pushes both
wells in the same direction, so the +/- response deltas would cancel under folding.

Risk-1 fix: the fold sign is the majority vote (sign of the time-averaged
Re(psi_c)) over the stationary window [pulse_idx:], not the single final snapshot,
so an isolated late inter-well hop cannot flip a trajectory's whole-history label.
"""

import numpy as np
import jax
import jax.numpy as jnp

from dtwa_non_integrated import (
    solve_single_trajectory,
    compute_explicit_bath_kernels,
    precompute_solver_arrays
)

# =====================================================================
# 1. TIME-DOMAIN CORRELATION ENGINE (O(N) Memory)
# =====================================================================

@jax.jit
def compute_exact_lag_correlation(x_batch):
    """Sum over a batch the unnormalized lagged products for every lag tau.

    For each lag tau in [0, N), returns sum over trajectories and over valid time
    origins t of x(t) * x(t + tau). The per-lag sample count (N - tau) is *not*
    divided out here; the caller normalizes (see :func:`calculate_correlations_and_responses`).
    Implemented with a ``lax.scan`` over lags to keep memory at O(N) rather than
    forming the full (N, N) outer product.

    Parameters
    ----------
    x_batch : float array, shape (batch_size, N) or (N,)
        Trajectories of a single real observable; 1-D input is promoted to 2-D.

    Returns
    -------
    corr : float array, shape (N,)
        ``corr[tau] = sum_{traj} sum_{t < N - tau} x(t) x(t + tau)``.
    """
    x_2d = jnp.atleast_2d(x_batch)
    N = x_2d.shape[1]
    lags = jnp.arange(N)

    def scan_body(carry, tau):
        shifted = jnp.roll(x_2d, -tau, axis=-1)
        valid_mask = jnp.arange(N) < (N - tau)
        c_tau = jnp.sum(x_2d * shifted * valid_mask)
        return carry, c_tau

    _, corr = jax.lax.scan(scan_body, None, lags)
    return corr

# =====================================================================
# 2. COMPILED MASTER ENGINE
# =====================================================================

@jax.jit(static_argnames=['pulse_idx'])
def _accumulate_batch_sums(spin_ensemble, cavity_ensemble, j_val, pulse_idx):
    """Reduce one batch into the raw/folded sums needed for C(tau) and responses.

    Maintains two parallel "tracks":

    * RAW track (for linear response): un-folded sums of S_x and Re(psi_c). The
      perturbation shifts both wells the same way, so the response delta survives
      the 50/50 well cancellation only if the trajectories are *not* folded.
    * FOLDED track (for correlations and means): each trajectory is mapped into the
      positive Z2 well so the macroscopic broken-symmetry baseline does not cancel.
      The fold sign is the majority vote (sign of the time-averaged Re(psi_c)) over
      the stationary window [pulse_idx:] -- robust to a single late inter-well hop.

    Parameters
    ----------
    spin_ensemble : float64 array, shape (batch_size, num_steps, 3)
        Spin trajectories.
    cavity_ensemble : complex128 array, shape (batch_size, num_steps)
        Cavity-amplitude trajectories.
    j_val : float
        Collective spin length j = N / 2.
    pulse_idx : int (static)
        Start index of the stationary window; correlations are computed on
        ``[pulse_idx:]`` (length N_tau = num_steps - pulse_idx).

    Returns
    -------
    dict of arrays
        ``sum_sx_raw``, ``sum_ra_raw``       : raw sums, shape (num_steps,)
        ``sum_sx_folded``, ``sum_ra_folded`` : folded sums, shape (num_steps,)
        ``sum_corr_sx``, ``sum_corr_ra``     : summed lagged products on the
                                               folded stationary window, shape (N_tau,)
    """
    sx_trajs = spin_ensemble[:, :, 0] / j_val
    r_alpha_trajs = jnp.real(cavity_ensemble)

    # ==========================================================
    # TRACK 1: RAW TRAJECTORIES (For Linear Response)
    # The perturbation pushes both wells in the same direction,
    # so the response delta survives the 50/50 cancellation.
    # ==========================================================
    sum_sx_raw = jnp.sum(sx_trajs, axis=0)
    sum_ra_raw = jnp.sum(r_alpha_trajs, axis=0)

    # ==========================================================
    # TRACK 2: FOLDED TRAJECTORIES (For C(tau) and Means)
    # We fold them to prevent the macroscopic mean field
    # from cancelling out and ruining the correlation baseline.
    #
    # Risk-1 fix: classify each trajectory's well by a MAJORITY VOTE over the
    # stationary window (sign of the time-averaged Re(psi_c) on [pulse_idx:]),
    # not the single final snapshot. A late isolated hop no longer flips the
    # whole-history label.
    # ==========================================================
    mean_ra_stat = jnp.mean(r_alpha_trajs[:, pulse_idx:], axis=1)
    signs = jnp.sign(mean_ra_stat)
    signs = jnp.where(signs == 0, 1.0, signs)[:, None]

    sx_folded = sx_trajs * signs
    ra_folded = r_alpha_trajs * signs

    # Slice the stationary part for exact lag correlation
    sx_stat_folded = sx_folded[:, pulse_idx:]
    ra_stat_folded = ra_folded[:, pulse_idx:]

    sum_corr_sx_folded = compute_exact_lag_correlation(sx_stat_folded)
    sum_corr_ra_folded = compute_exact_lag_correlation(ra_stat_folded)

    return {
        "sum_sx_raw": sum_sx_raw,
        "sum_ra_raw": sum_ra_raw,
        "sum_sx_folded": jnp.sum(sx_folded, axis=0),
        "sum_ra_folded": jnp.sum(ra_folded, axis=0),
        "sum_corr_sx": sum_corr_sx_folded,
        "sum_corr_ra": sum_corr_ra_folded
    }

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling', 'pulse_idx'])
def _compiled_master_processor(
    batched_keys,
    omega_0, B_field_safe, g, n_photons_initial, initial_direction,
    n_spins, dt, num_steps,
    Sigma_R_t, t_grid, w_pos, amp,
    use_noise, use_sampling,
    pulse_idx, epsilon_spin, epsilon_cavity
):
    """vmap+scan driver accumulating the raw/folded sums for one full ensemble.

    Vectorizes :func:`~dtwa_non_integrated.solve_single_trajectory` across each
    batch, reduces with :func:`_accumulate_batch_sums`, and sums over batches via
    ``lax.scan``. Called once per pass (base, spin-kick, cavity-kick).

    Parameters
    ----------
    batched_keys : PRNGKey array, shape (n_batches, batch_size, 2)
        Pre-batched RNG keys (identical across passes for paired-key response).
    omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins, dt, num_steps,
    Sigma_R_t, t_grid, w_pos, amp, use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity
        Forwarded to the trajectory solver (see that function). ``n_spins``,
        ``num_steps``, ``use_noise``, ``use_sampling`` and ``pulse_idx`` are static.

    Returns
    -------
    dict of arrays
        The summed statistics from :func:`_accumulate_batch_sums`, totalled across
        all trajectories.
    """

    j_val = n_spins / 2.0
    N_tau = num_steps - pulse_idx

    vmap_solver = jax.vmap(
        solve_single_trajectory,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry_stats, current_batch_keys):

        batch_S, batch_alpha = vmap_solver(
            current_batch_keys,
            omega_0, B_field_safe, g, n_photons_initial, initial_direction,
            n_spins, dt, num_steps,
            Sigma_R_t, t_grid, w_pos, amp,
            use_noise, use_sampling,
            pulse_idx, epsilon_spin, epsilon_cavity
        )

        batch_sums = _accumulate_batch_sums(batch_S, batch_alpha, j_val, pulse_idx)
        next_carry = {key: carry_stats[key] + batch_sums[key] for key in carry_stats}
        return next_carry, None

    init_stats = {
        "sum_sx_raw": jnp.zeros(num_steps),
        "sum_ra_raw": jnp.zeros(num_steps),
        "sum_sx_folded": jnp.zeros(num_steps),
        "sum_ra_folded": jnp.zeros(num_steps),
        "sum_corr_sx": jnp.zeros(N_tau),
        "sum_corr_ra": jnp.zeros(N_tau)
    }

    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats

# =====================================================================
# 3. SPECTRAL TRANSFORMS
# =====================================================================

@jax.jit
def compute_spectra(C_tau, chi_tau, dt, w_grid, eta=0.01):
    """Fourier-transform C(tau) and chi(tau) into S(omega) and Im chi(omega).

    Uses a one-sided transform with: an artificial broadening exp(-eta tau)
    (Lorentzian smoothing / regularization), a half-window cosine taper to
    suppress truncation ringing, Simpson quadrature weights, and DC removal
    (subtracting the tail mean) to kill spurious zero-frequency weight.

    Parameters
    ----------
    C_tau : float array, shape (N,)
        Connected correlation function on the stationary lag grid.
    chi_tau : float array, shape (N,)
        Linear-response function on the same grid.
    dt : float
        Lag spacing.
    w_grid : float array, shape (N_w,)
        Output angular-frequency grid.
    eta : float, optional
        Spectral broadening (inverse decay time of the smoothing kernel).

    Returns
    -------
    S_w : float array, shape (N_w,)
        Noise power spectrum, ``2 Re ∫ C(tau) e^{-i(w - i eta) tau} dtau``.
    neg_im_chi_w : float array, shape (N_w,)
        Dissipative response ``-Im chi(omega)``.
    """
    N = len(C_tau)
    tau_grid = jnp.arange(N) * dt

    # This transforms exp(-1j * w * t) into exp(-1j * w * t) * exp(-eta * t)
    exp_kernel = jnp.exp(-1j * (w_grid[:, None] - 1j * eta) * tau_grid[None, :])

    taper_start = int(0.5 * N)
    taper = jnp.ones(N)
    cos_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * (jnp.arange(N - taper_start) / (N - taper_start))))
    taper = taper.at[taper_start:].set(cos_taper)

    idx = jnp.arange(N)
    N_simpson = N - (1 - (N % 2))

    simpson_weights = jnp.where(
        idx < N_simpson,
        jnp.where(idx == 0, 1/3,
        jnp.where(idx == N_simpson - 1, 1/3,
        jnp.where(idx % 2 == 1, 4/3, 2/3))),
        0.0
    )

    is_even = (N % 2 == 0)
    simpson_weights += jnp.where(is_even & (idx == N - 2), 0.5, 0.0)
    simpson_weights += jnp.where(is_even & (idx == N - 1), 0.5, 0.0)

    weights = simpson_weights * taper

    tail_len = int(0.1 * N)
    C_tau = C_tau - jnp.mean(C_tau[-tail_len:])
    chi_tau = chi_tau - jnp.mean(chi_tau[-tail_len:])

    S_w = 2.0 * jnp.real(jnp.dot(exp_kernel, C_tau * weights) * dt)
    chi_w = jnp.dot(exp_kernel, chi_tau * weights) * dt

    return S_w, -jnp.imag(chi_w)

# =====================================================================
# 4. MAIN DRIVER
# =====================================================================

def calculate_correlations_and_responses(keys, t_grid, p, t_pulse, epsilon=1e-5, w_max=20.0, N_w=5000):
    """Measure C(tau), chi(tau) and their spectra for spin and cavity observables.

    Runs three ensembles that share the same RNG keys -- a base run, a spin-kicked
    run, and a cavity-kicked run -- and from them extracts the connected
    correlations (folded track) and the linear-response functions (raw track,
    via paired-key finite differences), then Fourier-transforms both to test the
    fluctuation-dissipation theorem.

    Parameters
    ----------
    keys : PRNGKey array, shape (n_total, 2)
        One key per trajectory (reused across all three passes).
    t_grid : float64 array, shape (num_steps,)
        Uniform time grid.
    p : dict
        Physical parameters. Required keys: ``omega_0``, ``alpha``, ``omega_c``,
        ``s``, ``T``, ``g``, ``n_photons_initial``, ``initial_direction``,
        ``n_spins``, ``batch_size``; optional ``B_z`` (default 0.0).
    t_pulse : float
        Time at which the impulsive perturbation is applied; also the start of the
        stationary window used for correlations (mapped to ``pulse_idx``).
    epsilon : float, optional
        Perturbation strength for the linear-response kicks.
    w_max : float, optional
        Upper edge of the output frequency grid.
    N_w : int, optional
        Number of output frequency points (geometric grid).

    Returns
    -------
    dict of numpy.ndarray
        ``tau_grid``                       : stationary lag grid, shape (N_tau,)
        ``w_grid``                         : output frequency grid, shape (N_w,)
        ``C_spin``, ``C_cavity``           : connected correlations C(tau)
        ``response_spin``, ``response_cavity`` : response functions chi(tau)
        ``S_c_spin``, ``S_c_cavity``       : noise power spectra S(omega)
        ``S_chi_spin``, ``S_chi_cavity``   : dissipative responses -Im chi(omega)
    """

    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    pulse_idx = int(np.searchsorted(t_grid, t_pulse))
    j_val = p['n_spins'] / 2.0
    N_tau = num_steps - pulse_idx

    B_base = jnp.zeros((num_steps, 3)).at[:, 2].set(p.get('B_z', 0.0))

    # ✅ NEW KERNEL PIPELINE
    Sigma_R_t, amp_full, w_grid_full, dw = compute_explicit_bath_kernels(
        num_steps, dt,
        p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'],
        w_max, N_w
    )

    Sigma_R_t, t_grid_pre, w_pos, amp = precompute_solver_arrays(
        num_steps, dt, Sigma_R_t, amp_full, dw, w_grid_full
    )

    batched_keys = keys[:(n_total // p['batch_size']) * p['batch_size']].reshape(-1, p['batch_size'], 2)

    print("Pass 1/3: Base...")
    res_base = _compiled_master_processor(
        batched_keys,
        p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'],
        p['n_spins'], dt, num_steps,
        Sigma_R_t, t_grid_pre, w_pos, amp,
        True, True,
        pulse_idx, 0.0, 0.0
    )

    print("Pass 2/3: Spin perturbation...")
    res_pert_spin = _compiled_master_processor(
        batched_keys,
        p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'],
        p['n_spins'], dt, num_steps,
        Sigma_R_t, t_grid_pre, w_pos, amp,
        True, True,
        pulse_idx, epsilon, 0.0
    )

    print("Pass 3/3: Cavity perturbation...")
    res_pert_cav = _compiled_master_processor(
        batched_keys,
        p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'],
        p['n_spins'], dt, num_steps,
        Sigma_R_t, t_grid_pre, w_pos, amp,
        True, True,
        pulse_idx, 0.0, epsilon
    )

    # === Correlations (Use FOLDED track) ===
    # We use the folded means to correctly subtract the massive broken-symmetry baseline
    mean_sx_folded = res_base["sum_sx_folded"] / n_total
    mean_ra_folded = res_base["sum_ra_folded"] / n_total

    mean_sx_stat_folded = mean_sx_folded[pulse_idx:]
    mean_ra_stat_folded = mean_ra_folded[pulse_idx:]

    mean_sx_corr = compute_exact_lag_correlation(mean_sx_stat_folded)
    mean_ra_corr = compute_exact_lag_correlation(mean_ra_stat_folded)

    counts = N_tau - jnp.arange(N_tau)
    counts = jnp.where(counts > 0, counts, 1.0)

    C_spin = ((res_base["sum_corr_sx"] / n_total) - mean_sx_corr) / counts
    C_cavity = (4.0 / j_val) * ((res_base["sum_corr_ra"] / n_total) - mean_ra_corr) / counts

    # === Responses (Use RAW track) ===
    # We use the raw, un-folded sums so the +/- response deltas don't cancel each other out
    response_spin = ((res_pert_spin["sum_sx_raw"] - res_base["sum_sx_raw"]) / n_total)[pulse_idx:] / (epsilon * j_val)
    response_cavity = 2.0 * ((res_pert_cav["sum_ra_raw"] - res_base["sum_ra_raw"]) / n_total)[pulse_idx:] / (epsilon * j_val)

    # === Spectra ===
    tau_grid = np.arange(N_tau) * dt
    w_min = 2.0 * jnp.pi / (N_tau * dt)
    w_grid = jnp.geomspace(w_min, w_max, N_w)

    print("Fourier transforms...")
    S_c_spin, S_chi_spin = compute_spectra(np.array(C_spin), np.array(response_spin), dt, w_grid, eta=1e-4)
    S_c_cavity, S_chi_cavity = compute_spectra(np.array(C_cavity), np.array(response_cavity), dt, w_grid, eta=1e-3)

    print("Done!")

    return {
        "tau_grid": tau_grid,
        "w_grid": np.array(w_grid),
        "C_spin": np.array(C_spin),
        "C_cavity": np.array(C_cavity),
        "response_spin": np.array(response_spin),
        "response_cavity": np.array(response_cavity),
        "S_c_spin": np.array(S_c_spin),
        "S_chi_spin": np.array(S_chi_spin),
        "S_c_cavity": np.array(S_c_cavity),
        "S_chi_cavity": np.array(S_chi_cavity)
    }
