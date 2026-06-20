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

import os
# Enable 64-bit precision BEFORE jax is imported, so importing this module first
# (e.g. ``from fdt import ...``) cannot silently leave the simulation in float32.
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_LEVEL"] = "error"

import numpy as np
import jax
import jax.numpy as jnp

from dtwa_non_integrated import (
    solve_single_trajectory,
    _resolve_bath_config,
)

# Minimum fraction of single-well trajectories below which post-selection is
# considered inapplicable (normal phase / unstable well) and is disabled. A
# genuine superradiant ensemble keeps a sizeable fraction even with hopping
# (e.g. ~50% at N=50); only the normal phase collapses toward 0%.
MIN_KEEP_FRACTION = 0.10

# =====================================================================
# 1. TIME-DOMAIN CORRELATION ENGINE (O(N) Memory)
# =====================================================================

@jax.jit
def compute_exact_lag_correlation(x_batch: jax.Array) -> jax.Array:
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
def _accumulate_batch_sums(spin_ensemble: jax.Array, cavity_ensemble: jax.Array,
                           j_val: float, pulse_idx: int,
                           traj_mask: jax.Array) -> dict:
    """Reduce one batch into the raw/folded sums needed for C(tau) and responses.

    Maintains two parallel "tracks":

    * RAW track (for linear response): un-folded sums of S_x and Re(psi_c). The
      perturbation shifts both wells the same way, so the response delta survives
      the 50/50 well cancellation only if the trajectories are *not* folded.
    * FOLDED track (for correlations and means): each trajectory is mapped into the
      positive Z2 well so the macroscopic broken-symmetry baseline does not cancel.
      The fold sign is the majority vote (sign of the time-averaged Re(psi_c)) over
      the stationary window [pulse_idx:] -- robust to a single late inter-well hop.

    Post-selection
    --------------
    ``traj_mask`` (per-trajectory 1.0 keep / 0.0 drop) restricts every reduction to
    a chosen sub-ensemble. In the superradiant phase it carries the single-well
    mask (trajectories that never cross the symmetry-breaking barrier during the
    measurement window), so the correlation and response are sampled in one well
    -- exactly as the broken-symmetry theory assumes -- and the finite-N inter-well
    hopping that otherwise corrupts the SR FDT is removed. Dropped trajectories
    contribute zero to every sum; the caller divides by ``n_kept`` (= sum of the
    mask), not by the raw trajectory count.

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
    traj_mask : float64 array, shape (batch_size,)
        Per-trajectory keep/drop weight (1.0 / 0.0). Pass all-ones to disable
        post-selection.

    Returns
    -------
    dict of arrays
        ``sum_sx_raw``, ``sum_ra_raw``       : raw sums, shape (num_steps,)
        ``sum_sx_folded``, ``sum_ra_folded`` : folded sums, shape (num_steps,)
        ``sum_corr_sx``, ``sum_corr_ra``     : summed lagged products on the
                                               folded stationary window, shape (N_tau,)
        ``n_kept``                           : scalar, number of retained trajectories
    """
    sx_trajs = spin_ensemble[:, :, 0] / j_val
    sy_trajs = spin_ensemble[:, :, 1] / j_val   # transverse spin: the Gaussian mode in SR
    r_alpha_trajs = jnp.real(cavity_ensemble)

    m = traj_mask[:, None]               # (batch, 1) broadcast weight
    n_kept = jnp.sum(traj_mask)

    # ==========================================================
    # TRACK 1: RAW TRAJECTORIES (For Linear Response)
    # The perturbation pushes both wells in the same direction,
    # so the response delta survives the 50/50 cancellation.
    # Masked: only the retained (single-well) trajectories contribute.
    # ==========================================================
    sum_sx_raw = jnp.sum(sx_trajs * m, axis=0)
    sum_sy_raw = jnp.sum(sy_trajs * m, axis=0)
    sum_ra_raw = jnp.sum(r_alpha_trajs * m, axis=0)

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
    sy_folded = sy_trajs * signs   # S_y is also Z2-odd, so it folds with the well sign

    # Slice the stationary window, then subtract the (masked) ensemble mean BEFORE
    # forming the lag products. This computes the connected correlation as
    # <(x - <x>)(x' - <x'>)> directly, avoiding the catastrophic cancellation
    # <x x'> - <x><x'> that wrecks precision when the mean is large (the SR-phase
    # coherent baseline |alpha|^2). The mean is the post-selected mean (kept
    # trajectories only); dropped trajectories are zeroed so they add nothing to
    # the lag products. Exact for a single batch (batch_size = n_total).
    sx_stat = sx_folded[:, pulse_idx:]
    sy_stat = sy_folded[:, pulse_idx:]
    ra_stat = ra_folded[:, pulse_idx:]
    inv_kept = 1.0 / (n_kept + 1e-12)
    mean_sx = jnp.sum(sx_stat * m, axis=0, keepdims=True) * inv_kept
    mean_sy = jnp.sum(sy_stat * m, axis=0, keepdims=True) * inv_kept
    mean_ra = jnp.sum(ra_stat * m, axis=0, keepdims=True) * inv_kept
    sx_fluc = (sx_stat - mean_sx) * m
    sy_fluc = (sy_stat - mean_sy) * m
    ra_fluc = (ra_stat - mean_ra) * m

    sum_corr_sx_folded = compute_exact_lag_correlation(sx_fluc)
    sum_corr_sy_folded = compute_exact_lag_correlation(sy_fluc)
    sum_corr_ra_folded = compute_exact_lag_correlation(ra_fluc)

    return {
        "sum_sx_raw": sum_sx_raw,
        "sum_sy_raw": sum_sy_raw,
        "sum_ra_raw": sum_ra_raw,
        "sum_sx_folded": jnp.sum(sx_folded * m, axis=0),
        "sum_ra_folded": jnp.sum(ra_folded * m, axis=0),
        "sum_corr_sx": sum_corr_sx_folded,
        "sum_corr_sy": sum_corr_sy_folded,
        "sum_corr_ra": sum_corr_ra_folded,
        "n_kept": n_kept,
    }

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling', 'pulse_idx',
                          'post_select', 'use_external_mask',
                          'cav_markovian', 'cav_rwa', 'rwa_interaction',
                          'spin_bath_on', 'spin_markovian', 'spin_axis'])
def _compiled_master_processor(
    batched_keys: jax.Array, ext_mask_batched: jax.Array,
    omega_0: float, B_field_safe: jax.Array, g: float, n_photons_initial: complex, initial_direction: jax.Array,
    n_spins: int, dt: float, num_steps: int,
    Sigma_R_cav: jax.Array, Sigma_R_spin: jax.Array, cos_wt: jax.Array, sin_wt: jax.Array,
    amp_cav: jax.Array, amp_spin: jax.Array,
    kappa_cav: float, kappa_spin: float, white_D_cav: float, white_D_spin: float, Szeq: float,
    use_noise: bool, use_sampling: bool,
    pulse_idx: int, epsilon_spin: float, epsilon_cavity: float, epsilon_spin_y: float = 0.0,
    post_select: bool = False, use_external_mask: bool = False,
    cav_markovian: bool = False, cav_rwa: bool = False, rwa_interaction: bool = False,
    spin_bath_on: bool = False, spin_markovian: bool = False, spin_axis: int = 0,
) -> tuple:
    """vmap+scan driver accumulating the raw/folded sums for one full ensemble.

    Vectorizes :func:`~dtwa_non_integrated.solve_single_trajectory` across each
    batch, reduces with :func:`_accumulate_batch_sums`, and sums over batches via
    ``lax.scan``. Called once per pass (base, spin-kick, cavity-kick).

    Post-selection (single-well mask)
    ---------------------------------
    Each trajectory is flagged "single-well" if its spin S_x keeps a constant sign
    across the whole measurement window [pulse_idx:] -- i.e. it never crosses the
    Z2 symmetry-breaking barrier (S_x = 0). In the superradiant phase the well sits
    far from zero (|S_x|/j ~ 0.8), so a sign change is an unambiguous inter-well
    hop; a clean trajectory's macroscopic order parameter never approaches zero.

    * ``post_select=True``, ``use_external_mask=False`` (base pass): the internal
      single-well flag is used as the reduction mask, and the per-trajectory flags
      are returned so the kicked passes can reuse them.
    * ``use_external_mask=True`` (kicked passes): the externally supplied mask
      (the base pass's flags) is used verbatim, so the paired-key finite-difference
      response is sampled over exactly the same trajectories as the base mean.
    * ``post_select=False`` (default): the mask is all-ones (no selection).

    Parameters
    ----------
    batched_keys : PRNGKey array, shape (n_batches, batch_size, 2)
        Pre-batched RNG keys (identical across passes for paired-key response).
    ext_mask_batched : float64 array, shape (n_batches, batch_size)
        External per-trajectory keep/drop mask, used only when
        ``use_external_mask`` is True (pass ones otherwise).
    post_select, use_external_mask : bool (static)
        Select the masking mode (see above).
    omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins, dt, num_steps,
    Sigma_R_t, cos_wt, sin_wt, amp, use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y
        Forwarded to the trajectory solver (see that function). ``n_spins``,
        ``num_steps``, ``use_noise``, ``use_sampling`` and ``pulse_idx`` are static.

    Returns
    -------
    (stats, clean_masks) : (dict of arrays, float64 array)
        ``stats``       : the summed statistics from :func:`_accumulate_batch_sums`,
                          totalled across all trajectories (includes ``n_kept``).
        ``clean_masks`` : per-trajectory single-well flags, shape
                          (n_batches, batch_size) -- the mask used (base pass) or
                          recomputed (otherwise), for the caller to reuse.
    """

    j_val = n_spins / 2.0
    N_tau = num_steps - pulse_idx

    def solve_one(key):
        return solve_single_trajectory(
            key, omega_0, B_field_safe, g, n_photons_initial, initial_direction,
            n_spins, dt, num_steps,
            Sigma_R_cav, Sigma_R_spin, cos_wt, sin_wt, amp_cav, amp_spin,
            kappa_cav, kappa_spin, white_D_cav, white_D_spin, Szeq,
            use_noise, use_sampling,
            pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y,
            cav_markovian, cav_rwa, rwa_interaction,
            spin_bath_on, spin_markovian, spin_axis)

    vmap_solver = jax.vmap(solve_one)

    def master_scan_body(carry_stats, scan_inputs):
        current_batch_keys, batch_ext_mask = scan_inputs

        batch_S, batch_alpha = vmap_solver(current_batch_keys)

        # Single-well flag: S_x keeps a constant sign over the measurement window.
        sx_stat = batch_S[:, pulse_idx:, 0]
        all_pos = jnp.all(sx_stat > 0.0, axis=1)
        all_neg = jnp.all(sx_stat < 0.0, axis=1)
        clean = (all_pos | all_neg).astype(batch_S.dtype)

        if use_external_mask:
            traj_mask = batch_ext_mask
        elif post_select:
            traj_mask = clean
        else:
            traj_mask = jnp.ones_like(clean)

        batch_sums = _accumulate_batch_sums(batch_S, batch_alpha, j_val, pulse_idx, traj_mask)
        next_carry = {key: carry_stats[key] + batch_sums[key] for key in carry_stats}
        return next_carry, clean

    init_stats = {
        "sum_sx_raw": jnp.zeros(num_steps),
        "sum_sy_raw": jnp.zeros(num_steps),
        "sum_ra_raw": jnp.zeros(num_steps),
        "sum_sx_folded": jnp.zeros(num_steps),
        "sum_ra_folded": jnp.zeros(num_steps),
        "sum_corr_sx": jnp.zeros(N_tau),
        "sum_corr_sy": jnp.zeros(N_tau),
        "sum_corr_ra": jnp.zeros(N_tau),
        "n_kept": jnp.zeros(()),
    }

    final_running_stats, clean_masks = jax.lax.scan(
        master_scan_body, init_stats, (batched_keys, ext_mask_batched))
    return final_running_stats, clean_masks

# =====================================================================
# 3. SPECTRAL TRANSFORMS
# =====================================================================

@jax.jit(static_argnames=['taper_frac'])
def compute_spectra(C_tau: jax.Array, chi_tau: jax.Array, dt: float,
                    w_grid: jax.Array, eta: float = 0.01, taper_frac: float = 0.5) -> tuple:
    """Fourier-transform C(tau) and chi(tau) into S(omega) and Im chi(omega).

    Uses a one-sided transform with: an artificial broadening exp(-eta tau)
    (Lorentzian smoothing / regularization), a Tukey cosine taper to suppress
    truncation ringing and spectral leakage, Simpson quadrature weights, and DC
    removal (subtracting the tail mean) to kill spurious zero-frequency weight.

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
    taper_frac : float, optional
        Tukey-window flat fraction: the leading ``taper_frac`` of the lag axis is
        left untapered and the remaining ``1 - taper_frac`` is rolled off by a
        raised cosine (always 1 at tau=0 to preserve the variance, 0 at the end).
        ``0.5`` (default) tapers only the last half; ``0.0`` tapers the whole
        range for maximum sidelobe suppression -- use it for gapped observables
        (e.g. the spin) whose strong peak otherwise leaks into the empty
        low-frequency band.

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

    # Tukey taper: flat over the leading taper_frac, raised-cosine roll-off after
    # (1 at tau=0, 0 at tau_max). Wider roll-off -> lower spectral-leakage floor.
    taper_start = int(taper_frac * N)
    n_dec = max(N - taper_start, 1)
    taper = jnp.ones(N)
    cos_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * (jnp.arange(n_dec) / max(n_dec - 1, 1))))
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

def calculate_correlations_and_responses(keys: jax.Array, t_grid: jax.Array, p: dict, t_pulse: float,
                                         epsilon: float = 1e-5, w_max: float = 20.0, N_w: int = 5000,
                                         mem_window: int = None, post_select: bool = False,
                                         eta_spin: float = 5e-2, eta_cavity: float = 1e-3) -> dict:
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
    mem_window : int or None, optional
        [P1] Retarded-kernel truncation length. ``None`` (default) keeps the full
        history (exact); an integer L truncates the memory integral to the last L
        steps. Validate by confirming the spectra are unchanged as L grows.
    post_select : bool, optional
        If True, keep only single-well trajectories (S_x never crosses zero during
        the measurement window) when forming the correlations and responses. The
        keep-mask is computed on the base pass and reused verbatim on the kicked
        passes, so the paired-key response stays consistent. This removes the
        finite-N inter-well hopping that corrupts the superradiant-phase FDT,
        giving clean single-well spectra. Only meaningful in the superradiant phase
        (where the order parameter is macroscopic and sits far from zero); in the
        normal phase S_x fluctuates about zero, so leave it False there.

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

    B_z_val = p.get('B_z', 0.0)
    B_base = jnp.zeros((num_steps, 3)).at[:, 2].set(B_z_val)

    # --- Bath / interaction selection (read from p, defaulting to the original
    #     cavity-only, non-Markovian, full-coupling, Dicke setup) ---
    cav_markovian = bool(p.get('cavity_bath_markovian', False))
    cav_rwa = bool(p.get('rwa_cavity_bath', False))
    rwa_interaction = bool(p.get('rwa_interaction', False))
    spin_markovian = bool(p.get('spin_bath_markovian', False))

    # Build both bath kernels + shared trig matrices.
    cfg = _resolve_bath_config(
        p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], B_z_val,
        p['n_spins'], dt, num_steps, w_max, N_w,
        cav_markovian, cav_rwa,
        p.get('alpha_spin', 0.0), p.get('omega_c_spin', None), p.get('s_spin', None),
        p.get('T_spin', None), spin_markovian, p.get('rwa_spin_bath', False),
        p.get('spin_bath_channel', 'transverse'))
    Sig_cav, Sig_spin = cfg["Sigma_R_cav"], cfg["Sigma_R_spin"]
    spin_bath_on, spin_axis = cfg["spin_bath_on"], cfg["spin_axis"]

    # [P1] Memory-window truncation (None -> full, exact). Shared by all passes.
    if not cav_markovian:
        mag = np.abs(np.asarray(Sig_cav)); mag = mag / (mag[0] + 1e-300)
        def _decay(tol):
            idx = int(np.argmax(mag < tol)); return idx if idx > 0 else num_steps
        L = num_steps if mem_window is None else int(min(mem_window, num_steps))
        Sig_cav = Sig_cav[:L]
        print(f"Memory window L={L}/{num_steps}  (|Sigma_R^cav| <1e-3 by step "
              f"{_decay(1e-3)}, <1e-4 by {_decay(1e-4)})")
    else:
        Sig_cav = Sig_cav[:1]
        print(f"Cavity bath: MARKOVIAN (kappa={cfg['kappa_cav']:.4g})")
    if spin_bath_on and not spin_markovian:
        Sig_spin = Sig_spin[:max(Sig_cav.shape[0], 1)]
    else:
        Sig_spin = Sig_spin[:1]

    n_used = (n_total // p['batch_size']) * p['batch_size']
    batched_keys = keys[:n_used].reshape(-1, p['batch_size'], 2)
    ones_mask = jnp.ones((batched_keys.shape[0], p['batch_size']))

    def _run(eps_spin, eps_cav, eps_spin_y, label, ext_mask, use_ext, ps_flag):
        print(label)
        return _compiled_master_processor(
            batched_keys, ext_mask,
            p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'],
            p['n_spins'], dt, num_steps,
            Sig_cav, Sig_spin, cfg["cos_wt"], cfg["sin_wt"], cfg["amp_cav"], cfg["amp_spin"],
            cfg["kappa_cav"], cfg["kappa_spin"], cfg["white_D_cav"], cfg["white_D_spin"], cfg["Szeq"],
            True, True,
            pulse_idx, eps_spin, eps_cav, eps_spin_y,
            ps_flag, use_ext,
            cav_markovian, cav_rwa, rwa_interaction,
            spin_bath_on, spin_markovian, spin_axis,
        )

    # Base pass computes the single-well keep-mask; the kicked passes reuse it
    # verbatim (use_ext=True) so the paired-key response is sampled over exactly
    # the same trajectories. With post_select=False the mask is all-ones (no-op).
    res_base, clean_masks = _run(0.0, 0.0, 0.0, "Pass 1/4: Base...", ones_mask, False, post_select)

    # Normal-phase / over-melting guard. Post-selection keeps only trajectories
    # whose S_x never changes sign; in the normal phase <S_x> = 0 so *every*
    # trajectory crosses zero, the kept count collapses to ~0, and dividing the
    # sums by n_kept would yield all-NaN spectra. If the kept fraction is too low,
    # the single-well assumption simply does not apply here -- fall back to the
    # full ensemble (no selection) and warn, rather than returning garbage.
    effective_ps = post_select
    n_kept = float(res_base["n_kept"])
    if post_select:
        frac = n_kept / max(n_used, 1)
        if frac < MIN_KEEP_FRACTION:
            print(f"WARNING: post-selection kept only {100.0 * frac:.1f}% of trajectories "
                  f"(< {100.0 * MIN_KEEP_FRACTION:.0f}%). The single-well assumption does not "
                  f"hold here (normal phase, or the broken-symmetry well is unstable at this "
                  f"T). Falling back to NO post-selection.")
            effective_ps = False
            res_base, clean_masks = _run(0.0, 0.0, 0.0, "Pass 1/4 (re-run, no post-select)...",
                                         ones_mask, False, False)
            n_kept = float(res_base["n_kept"])
        else:
            print(f"Post-selection: kept {int(n_kept)}/{n_used} single-well trajectories "
                  f"({100.0 * frac:.1f}%).")

    kick_mask = clean_masks if effective_ps else ones_mask
    res_pert_spin, _ = _run(epsilon, 0.0,     0.0,     "Pass 2/4: Spin (S_x) perturbation...", kick_mask, effective_ps, effective_ps)
    res_pert_cav, _  = _run(0.0,     epsilon, 0.0,     "Pass 3/4: Cavity perturbation...",     kick_mask, effective_ps, effective_ps)
    res_pert_sy, _   = _run(0.0,     0.0,     epsilon, "Pass 4/4: Transverse spin (S_y) perturbation...", kick_mask, effective_ps, effective_ps)

    # Number of retained trajectories (= n_used when post_select is off). All
    # correlations and responses are normalized by this, NOT the raw count, so the
    # post-selected sub-ensemble is averaged correctly. The base mask is reused on
    # every kicked pass, so n_kept is identical across passes.

    # === Correlations (FOLDED track) ===
    # The mean is already subtracted inside _accumulate_batch_sums (connected,
    # numerically stable), so here we just normalize by the per-lag sample count.
    counts = N_tau - jnp.arange(N_tau)
    counts = jnp.where(counts > 0, counts, 1.0)

    C_spin = (res_base["sum_corr_sx"] / n_kept) / counts
    C_spin_y = (res_base["sum_corr_sy"] / n_kept) / counts
    C_cavity = (4.0 / j_val) * (res_base["sum_corr_ra"] / n_kept) / counts

    # === Responses (Use RAW track) ===
    # We use the raw, un-folded sums so the +/- response deltas don't cancel each other out
    response_spin = ((res_pert_spin["sum_sx_raw"] - res_base["sum_sx_raw"]) / n_kept)[pulse_idx:] / (epsilon * j_val)
    response_spin_y = ((res_pert_sy["sum_sy_raw"] - res_base["sum_sy_raw"]) / n_kept)[pulse_idx:] / (epsilon * j_val)
    response_cavity = 2.0 * ((res_pert_cav["sum_ra_raw"] - res_base["sum_ra_raw"]) / n_kept)[pulse_idx:] / (epsilon * j_val)

    # === Spectra ===
    tau_grid = np.arange(N_tau) * dt
    w_min = 2.0 * jnp.pi / (N_tau * dt)
    w_grid = jnp.geomspace(w_min, w_max, N_w)

    print("Fourier transforms...")
    # S_x is gapped/pinned (especially in SR); a full-range Tukey taper
    # (taper_frac=0.0) suppresses spectral leakage. The transverse S_y is the
    # Gaussian mode in SR and carries the meaningful FDT signal there.
    #
    # The spin response is sharply peaked at the polaritons with ~zero weight
    # between, so a small eta leaves the FDT ratio defined only in narrow bands.
    # A larger ``eta_spin`` broadens the peaks (applied identically to S_c and
    # S_chi, so the ratio is preserved) until their tails overlap and the ratio
    # becomes a continuous curve across the spectrum. The cavity is intrinsically
    # broad, so it needs far less (``eta_cavity``).
    S_c_spin, S_chi_spin = compute_spectra(np.array(C_spin), np.array(response_spin), dt, w_grid,
                                           eta=eta_spin, taper_frac=0.0)
    S_c_spin_y, S_chi_spin_y = compute_spectra(np.array(C_spin_y), np.array(response_spin_y), dt, w_grid,
                                               eta=eta_spin, taper_frac=0.0)
    S_c_cavity, S_chi_cavity = compute_spectra(np.array(C_cavity), np.array(response_cavity), dt, w_grid,
                                               eta=eta_cavity, taper_frac=0.5)

    print("Done!")

    return {
        "tau_grid": tau_grid,
        "w_grid": np.array(w_grid),
        "C_spin": np.array(C_spin),
        "C_spin_y": np.array(C_spin_y),
        "C_cavity": np.array(C_cavity),
        "response_spin": np.array(response_spin),
        "response_spin_y": np.array(response_spin_y),
        "response_cavity": np.array(response_cavity),
        "S_c_spin": np.array(S_c_spin),
        "S_chi_spin": np.array(S_chi_spin),
        "S_c_spin_y": np.array(S_c_spin_y),
        "S_chi_spin_y": np.array(S_chi_spin_y),
        "S_c_cavity": np.array(S_c_cavity),
        "S_chi_cavity": np.array(S_chi_cavity)
    }
