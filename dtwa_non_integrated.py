"""
Discrete Truncated Wigner Approximation (DTWA) for the open Dicke model
with an *explicit*, non-integrated cavity mode.

This module implements the coupled semiclassical equations of motion derived in
the accompanying Keldysh notes ("Single Spin and Single Lossy Boson Model"):

    Cavity (non-Markovian generalized Langevin equation, Eq. 14):
        i d/dt psi_c(t) = omega_0 psi_c(t)
                          + ∫_{-inf}^{t} dt' Sigma_R(t - t') psi_c(t')
                          + (2 sqrt(2) g / sqrt(N)) S_x(t)
                          - xi(t)

    Spin (kinematic precession, Eqs. 15-16):
        d/dt S(t) = B_eff(t) x S(t),
        B_eff(t) = B + x_hat (2 sqrt(2) g / sqrt(N)) (psi_c(t) + psi_c*(t))

The cavity carries the bath memory kernel Sigma_R(t) (retarded self-energy) and
is driven by the colored stochastic noise xi(t) whose statistics are fixed by
the fluctuation-dissipation theorem, <xi(t) xi*(t')> = -i Sigma_K(t - t').

Pipeline
--------
1. ``compute_explicit_bath_kernels``  : build Sigma_R(t) and the noise amplitude
                                        spectrum from the Ohmic-family J(omega).
2. ``precompute_solver_arrays``       : slice the positive-frequency half used by
                                        the per-trajectory noise generator.
3. ``solve_single_trajectory``        : integrate one coupled spin+cavity
                                        trajectory with a Heun predictor-corrector.
4. ``_compiled_master_processor``     : vmap over a batch of trajectories and
                                        accumulate ensemble statistics.
5. ``run_dtwa``                       : top-level driver returning ensemble means.

Conventions
-----------
* Spins are stored as Cartesian vectors S = (S_x, S_y, S_z) in units where the
  full collective spin length is j = N / 2 (so intensive components are S / j).
* The cavity amplitude ``alpha`` is the complex coherent-state field psi_c.
* All real arrays are float64 and the cavity is complex128 (x64 is enabled at
  import time via the JAX_ENABLE_X64 environment flag).
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_LEVEL"] = "error"

import jax
import jax.numpy as jnp
import numpy as np
from initial_samplings import discrete_spin_sampling_factorized, cavity_wigner_sampling

# =====================================================================
# 1. KERNELS & PRECOMPUTE ENGINE
# =====================================================================

@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_explicit_bath_kernels(num_steps: int, dt: float, omega_0: float, alpha: float,
                                  omega_c: float, s: float, T: float,
                                  w_max: float = 40.0, N_w: int = 5000) -> tuple:
    """Build the retarded memory kernel and noise spectrum for the cavity bath.

    Full (non-RWA) position-coupling bath: the cavity couples to the bath via the
    X quadrature, (a + a^dag) sum_k g_k (b_k + b_k^dag), keeping the counter-rotating
    terms. The bath spectral density is the Ohmic-family form (notes Eq. 17),
        J(omega) = alpha * omega_c * (omega / omega_c)**s * exp(-omega / omega_c),  omega > 0,
    from which we compute:

    * the real Caldeira-Leggett friction kernel
          Sigma_R(t) = -2 theta(t) int_0^inf dw J(w) sin(w t),
      whose Fourier transform has Im Sigma_R(w) = -pi J(w) (dissipation) AND a
      nonzero real part Lambda(w) (the collective Lamb shift from the counter-
      rotating terms, which renormalizes omega_0 and shifts g_c). It acts on the
      X = psi_c + psi_c* quadrature.
    * the per-frequency noise amplitude sqrt( 2 J(w) coth(w / 2T) dw ) for the real
      bath force xi(t) on that quadrature, with target correlation
          <xi(t) xi(t')> = 2 int_0^inf dw J(w) coth(w / 2T) cos(w (t - t')),
      FDT-consistent with the friction kernel.

    Parameters
    ----------
    num_steps : int (static)
        Number of time steps; Sigma_R is tabulated on t = [0, dt, ..., (num_steps-1) dt].
    dt : float
        Time-step size.
    omega_0 : float
        Bare cavity frequency (passed through for interface symmetry; not used here).
    alpha : float
        Dimensionless system-bath coupling strength (the "alpha" of J(omega)).
    omega_c : float
        Bath high-frequency cutoff.
    s : float
        Ohmicity exponent: s < 1 sub-Ohmic, s = 1 Ohmic, s > 1 super-Ohmic.
    T : float
        Bath temperature (same energy units as the frequencies; kB = 1).
    w_max : float, optional
        Half-width of the symmetric frequency grid used for the FT integrals.
    N_w : int (static), optional
        Number of frequency grid points on [-w_max, w_max].

    Returns
    -------
    Sigma_R_t : complex128 array, shape (num_steps,)
        Real friction kernel (stored as complex128 for the solver) at the time grid.
    amp_full : float64 array, shape (N_w,)
        Noise amplitude sqrt(2 J(w) coth(w/2T) dw) on the *full* (signed) w grid.
    w_grid : float64 array, shape (N_w,)
        The signed frequency grid [-w_max, w_max].
    dw : float
        Frequency spacing of ``w_grid``.
    """
    w_grid = jnp.linspace(-w_max, w_max, N_w, dtype=jnp.float64)
    dw = w_grid[1] - w_grid[0]
    J_w_pos = jnp.where(w_grid > 1e-10, alpha * omega_c * (w_grid/omega_c)**s * jnp.exp(-w_grid/omega_c), 0.0)

    # [Full / non-RWA] Real Caldeira-Leggett friction kernel
    #     Sigma_R(t) = -2 theta(t) int_0^inf dw J(w) sin(w t).
    # J(w) > 0 only for w > 0, so the sin-transform over the full grid picks out the
    # positive-frequency integral. Im Sigma_R(w) = -pi J(w) (dissipation); its real
    # part is the Lamb shift Lambda(w) from the counter-rotating terms.
    t_grid = jnp.arange(num_steps) * dt
    Sigma_R_t = (-2.0 * jnp.dot(jnp.sin(t_grid[:, None] * w_grid[None, :]), J_w_pos) * dw
                 ).astype(jnp.complex128)

    # [A3] FDT-consistent real bath-noise amplitude on the X quadrature; the factor
    # 2 follows from S_xi(w) = coth(w/2T) * (-2 Im Sigma_R(w)). The where() guards
    # the 0 * inf -> NaN at w = 0 (coth diverges while J vanishes).
    coth_w = 1.0 / jnp.tanh(w_grid / (2.0 * T + 1e-12))
    amp_full = jnp.sqrt(jnp.where(J_w_pos > 0.0, 2.0 * J_w_pos * coth_w * dw, 0.0))

    return Sigma_R_t, amp_full, w_grid, dw


@jax.jit(static_argnames=['num_steps'])
def precompute_solver_arrays(num_steps: int, dt: float, Sigma_R_t: jax.Array,
                             amp_full: jax.Array, dw: float, w_grid: jax.Array) -> tuple:
    """Slice the positive-frequency half of the noise spectrum for trajectory use.

    The colored-noise synthesizer only needs the positive-frequency branch of the
    amplitude spectrum (the negative branch is redundant once xi is built from a
    real cosine/sine pair). This helper splits ``amp_full``/``w_grid`` at w = 0 and
    rebuilds the time grid.

    Parameters
    ----------
    num_steps : int (static)
        Number of time steps.
    dt : float
        Time-step size.
    Sigma_R_t : complex128 array, shape (num_steps,)
        Retarded kernel from :func:`compute_explicit_bath_kernels` (passed through).
    amp_full : float64 array, shape (N_w,)
        Full signed-frequency noise amplitude.
    dw : float
        Frequency spacing (unused here; kept for interface symmetry).
    w_grid : float64 array, shape (N_w,)
        Signed frequency grid.

    Returns
    -------
    Sigma_R_t : complex128 array, shape (num_steps,)
        Passed through unchanged.
    t_grid : float64 array, shape (num_steps,)
        Simulation time grid t = arange(num_steps) * dt.
    cos_wt : float64 array, shape (num_steps, N_w // 2)
        Precomputed cos(t * w) on the positive-frequency grid, shared by all
        trajectories so each colored-noise realization is a single GEMM.
    sin_wt : float64 array, shape (num_steps, N_w // 2)
        Precomputed sin(t * w) on the positive-frequency grid.
    amp : float64 array, shape (N_w // 2,)
        Noise amplitude on the positive-frequency grid.
    """
    half_N = amp_full.shape[0] // 2
    w_pos = w_grid[half_N:]
    amp = amp_full[half_N:]
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    # [P3] Build the shared time-frequency trig matrices ONCE. The colored noise
    # for every trajectory then reduces to (num_steps, half_N) @ (half_N,) GEMMs,
    # instead of recomputing cos/sin of a length-half_N vector at every step.
    wt = t_grid[:, None] * w_pos[None, :]
    cos_wt = jnp.cos(wt)
    sin_wt = jnp.sin(wt)
    return Sigma_R_t, t_grid, cos_wt, sin_wt, amp

# =====================================================================
# 2. HIGH-SPEED TRAJECTORY SOLVERS (PURE HEUN'S METHOD)
# =====================================================================

def non_markovian_coupled_heun_step(S_history: jax.Array, alpha_history: jax.Array, step_idx: int,
                                    xi_curr: jax.Array, xi_next: jax.Array, Sigma_R_t: jax.Array,
                                    B_field_val: jax.Array, coupling_strength: float,
                                    omega_0: float, dt: float) -> tuple:
    """Advance the coupled spin+cavity state by one Heun predictor-corrector step.

    The spin is rotated rigidly about the instantaneous effective field (Rodrigues
    rotation, exactly norm-preserving). The cavity is advanced with an
    integrating-factor ETDRK2 predictor-corrector ([A2]): the bare oscillation
    -i omega_0 psi is propagated exactly (no spurious energy drift), while the
    memory, spin drive and noise are handled to 2nd order. The retarded memory
    integral ∫ Sigma_R(t - t') psi_c(t') dt' is evaluated on the fly with
    trapezoidal weights over a truncated window of length L = Sigma_R_t.shape[0]
    ([P1]), so it costs O(L) per step instead of O(num_steps).

    Parameters
    ----------
    S_history : float64 array, shape (num_steps, 3)
        Spin trajectory buffer; rows [0 .. step_idx-1] are already filled.
    alpha_history : complex128 array, shape (num_steps,)
        Cavity trajectory buffer; entries [0 .. step_idx-1] are already filled.
    step_idx : int
        Index of the new step being computed (>= 1).
    xi_curr : complex128 scalar
        Bath noise xi(t) at the current time (t = (step_idx-1) dt).
    xi_next : complex128 scalar
        Bath noise xi(t) at the target time (t = step_idx dt).
    Sigma_R_t : complex128 array, shape (L,)
        Tabulated retarded memory kernel, truncated to the memory-window length L
        ([P1]); L == num_steps means the full, untruncated history.
    B_field_val : float64 array, shape (3,)
        External field B at this step.
    coupling_strength : float
        g / sqrt(N), the per-particle light-matter coupling.
    omega_0 : float
        Bare cavity frequency.
    dt : float
        Time-step size.

    Returns
    -------
    S_history : float64 array, shape (num_steps, 3)
        Buffer with row ``step_idx`` set to the new spin vector.
    alpha_history : complex128 array, shape (num_steps,)
        Buffer with entry ``step_idx`` set to the new cavity amplitude.

    Notes
    -----
    The B_eff x-component uses ``4 * coupling_strength * Re(alpha)`` and the cavity
    spin drive uses ``2 * coupling_strength * S_x``; these encode the
    S = (1/sqrt(2)) J_c convention of the notes.
    """
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    alpha_curr = alpha_history[curr_idx]

    # ==========================================================
    # --- [P1] WINDOWED TRAPEZOIDAL ROW GENERATOR ---
    # Evaluates the retarded convolution ∫ Sigma_R(t-t') alpha(t') dt' using only
    # the most recent L = Sigma_R_t.shape[0] history points (trapezoidal weights,
    # 0.5 at the endpoints, strict causality). Truncating to L turns the O(N^2)
    # history sum into O(N*L); L == num_steps reproduces the full convolution
    # exactly, since the kernel beyond the window is the part that has decayed.
    # ==========================================================
    L = Sigma_R_t.shape[0]
    k_idx = jnp.arange(L)

    def get_memory(target_idx, alpha_hist):
        start = jnp.maximum(target_idx - L + 1, 0)
        block = jax.lax.dynamic_slice(alpha_hist, (start,), (L,))
        block_j = start + k_idx
        lag = target_idx - block_j
        valid = lag >= 0
        safe_lag = jnp.where(valid, lag, 0)
        sigma_vals = jnp.where(valid, Sigma_R_t[safe_lag], 0.0j)

        weights = jnp.where(valid, 1.0, 0.0)
        weights = jnp.where(block_j == 0, 0.5, weights)
        weights = jnp.where(block_j == target_idx, 0.5, weights)
        weights = jnp.where(target_idx == 0, 0.0, weights)

        # [Full / non-RWA] the bath couples to the X quadrature x = psi_c + psi_c*,
        # so the memory integral acts on (block + conj(block)) = 2 Re(psi_c), not on
        # psi_c alone (the RWA approximation).
        x_quad = block + jnp.conj(block)
        return jnp.dot(sigma_vals * weights * dt, x_quad)
    # ==========================================================

    # --- [A2] Integrating-factor (ETDRK2) coefficients for dpsi/dt = -i w0 psi + N.
    # The bare oscillation is integrated EXACTLY (|E| = 1, no energy drift); a
    # 2nd-order exponential predictor-corrector handles the drive N. As w0 -> 0
    # these reduce to (E, phi1, b1, b2) -> (1, 1, dt/2, dt/2), i.e. plain Heun.
    z = -1j * omega_0 * dt
    E = jnp.exp(z)
    _small = jnp.abs(z) < 1e-8
    phi1 = jnp.where(_small, 1.0 + z / 2.0, (E - 1.0) / z)
    phi2 = jnp.where(_small, 0.5 + z / 6.0, (E - 1.0 - z) / (z * z))
    b1 = dt * (phi1 - phi2)
    b2 = dt * phi2

    # --- 1. PREDICTOR (Standard Explicit Euler) ---
    memory_p = get_memory(curr_idx, alpha_history)

    B_eff_p_x = B_field_val[0] + 4.0 * coupling_strength * jnp.real(alpha_curr)
    B_eff_p = jnp.array([B_eff_p_x, B_field_val[1], B_field_val[2]], dtype=jnp.float64)
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = b_mag_p * dt

    S_pred = (S_curr * jnp.cos(angle_p) +
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) +
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))

    drive_p = -1j * memory_p - 1j * 2.0 * coupling_strength * S_curr[0] + 1j * xi_curr

    # [A2] Exponential predictor (ETD exponential Euler): linear part exact.
    alpha_pred = E * alpha_curr + dt * phi1 * drive_p
    alpha_history_pred = alpha_history.at[step_idx].set(alpha_pred)

    # --- 2. CORRECTOR (Standard Explicit Trapezoidal) ---
    memory_c = get_memory(step_idx, alpha_history_pred)

    B_eff_c_x = B_field_val[0] + 4.0 * coupling_strength * jnp.real(alpha_pred)
    B_eff_c = jnp.array([B_eff_c_x, B_field_val[1], B_field_val[2]], dtype=jnp.float64)
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = b_mag_avg * dt

    S_next = (S_curr * jnp.cos(angle_avg) +
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) +
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))

    drive_c = -1j * memory_c - 1j * 2.0 * coupling_strength * S_pred[0] + 1j * xi_next

    # [A2] ETDRK2 corrector: exact linear propagator + 2nd-order drive.
    alpha_next = E * alpha_curr + b1 * drive_p + b2 * drive_c

    return S_history.at[step_idx].set(S_next), alpha_history.at[step_idx].set(alpha_next)


def solve_single_trajectory(key: jax.Array, omega_0: float, B_field_base: jax.Array, g: float,
                            alpha_shift: complex, initial_direction: jax.Array, n_spins: int,
                            dt: float, num_steps: int, Sigma_R_t: jax.Array, cos_wt: jax.Array,
                            sin_wt: jax.Array, amp: jax.Array, use_noise: bool, use_sampling: bool,
                            pulse_idx: int, epsilon_spin: float, epsilon_cavity: float,
                            epsilon_spin_y: float = 0.0) -> tuple:
    """Integrate a single coupled spin+cavity DTWA trajectory.

    Samples an initial spin (discrete Wigner) and cavity amplitude (Gaussian
    Wigner), synthesizes the colored bath noise xi(t), then time-steps the coupled
    equations with :func:`non_markovian_coupled_heun_step`. An optional impulsive
    perturbation ("pulse") can be applied at a single time index to measure linear
    response.

    Parameters
    ----------
    key : jax.random.PRNGKey
        RNG key for this trajectory (split internally into spin / cavity / noise).
    omega_0 : float
        Bare cavity frequency.
    B_field_base : float64 array, shape (num_steps, 3)
        External field at every time step.
    g : float
        Light-matter coupling (the per-particle strength is g / sqrt(n_spins)).
    alpha_shift : complex
        Mean initial cavity amplitude (coherent displacement).
    initial_direction : float64 array, shape (3,)
        Initial spin direction (need not be normalized).
    n_spins : int
        Number of two-level atoms N; the collective spin length is j = N / 2.
    dt : float
        Time-step size.
    num_steps : int
        Number of time steps.
    Sigma_R_t : complex128 array, shape (num_steps,)
        Retarded memory kernel.
    cos_wt : float64 array, shape (num_steps, N_w // 2)
        Shared cos(t * w) matrix for synthesizing the colored noise (see
        :func:`precompute_solver_arrays`).
    sin_wt : float64 array, shape (num_steps, N_w // 2)
        Shared sin(t * w) matrix for synthesizing the colored noise.
    amp : float64 array, shape (N_w // 2,)
        Noise amplitude spectrum on the positive-frequency grid.
    use_noise : bool
        If True, inject the bath noise xi(t); if False, xi = 0 (deterministic).
    use_sampling : bool
        If True, sample initial conditions from the Wigner distribution;
        if False, use the mean-field initial state.
    pulse_idx : int
        Time index at which to apply the impulsive perturbation (use a value
        outside [1, num_steps) to disable, e.g. -1).
    epsilon_spin : float
        Rotation angle of the impulsive spin kick about the x-axis (rotates y->z).
    epsilon_cavity : float
        Imaginary-quadrature kick added to the cavity amplitude at the pulse.

    Returns
    -------
    final_S_history : float64 array, shape (num_steps, 3)
        Full spin trajectory.
    final_alpha_history : complex128 array, shape (num_steps,)
        Full cavity-amplitude trajectory.
    """
    k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
    coupling_strength = g / jnp.sqrt(n_spins)

    s0_sampled = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins) / 2.0
    s0_mean = (initial_direction * n_spins) / 2.0
    s0 = jnp.where(use_sampling, s0_sampled, s0_mean)


    alpha0_mean = jnp.array(alpha_shift, dtype=jnp.complex128)
    alpha0_sampled = cavity_wigner_sampling(k_samp_alpha, alpha0_mean)
    alpha0 = jnp.where(use_sampling, alpha0_sampled, alpha0_mean)

    S_history = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[0].set(s0)
    alpha_history = jnp.zeros((num_steps,), dtype=jnp.complex128).at[0].set(alpha0)

    half_N = amp.shape[0]
    k_re, k_im = jax.random.split(k_noise)
    noise_re = jax.random.normal(k_re, (half_N,), dtype=jnp.float64) * amp
    noise_im = jax.random.normal(k_im, (half_N,), dtype=jnp.float64) * amp

    # [Full / non-RWA] Real symmetric bath force xi(t) on the X quadrature, one GEMM
    # against the shared trig matrices:
    #   xi(t) = sum_k amp_k [cos(w_k t) u_k + sin(w_k t) v_k],
    #   <xi(t) xi(t')> = 2 int_0^inf J(w) coth(w/2T) cos(w (t - t')) dw.
    xi_t = cos_wt @ noise_re + sin_wt @ noise_im
    xi_all = jnp.where(use_noise, xi_t.astype(jnp.complex128), 0.0)

    def scan_body(carry, step_idx):
        S_hist, alpha_hist = carry
        curr_idx = step_idx - 1

        B_val_step = jax.lax.dynamic_index_in_dim(B_field_base, curr_idx, axis=0, keepdims=False)
        xi_curr = xi_all[curr_idx]
        xi_next = xi_all[step_idx]

        # ETDRK2 (integrating-factor) coupled spin+cavity step over a memory window
        S_hist_updated, alpha_hist_updated = non_markovian_coupled_heun_step(
            S_hist, alpha_hist, step_idx, xi_curr, xi_next, Sigma_R_t,
            B_val_step, coupling_strength, omega_0, dt
        )

        current_S = S_hist_updated[step_idx]
        current_alpha = alpha_hist_updated[step_idx]

        # ==========================================================
        # EXACT ANALYTICAL JUMPS (impulsive linear-response perturbation)
        # ==========================================================
        is_pulse = (step_idx == pulse_idx)
        # x-rotation (impulsive B_x field): rotates y,z, leaves S_x -> probes chi_xx
        cx, sx = jnp.cos(epsilon_spin), jnp.sin(epsilon_spin)
        Sy1 = current_S[1] * cx - current_S[2] * sx
        Sz1 = current_S[2] * cx + current_S[1] * sx
        # y-rotation (impulsive B_y field): rotates x,z, leaves S_y -> probes chi_yy
        cy, sy = jnp.cos(epsilon_spin_y), jnp.sin(epsilon_spin_y)
        Sx2 = current_S[0] * cy + Sz1 * sy
        Sz2 = -current_S[0] * sy + Sz1 * cy
        S_jumped = jnp.array([Sx2, Sy1, Sz2])

        current_S = jnp.where(is_pulse, S_jumped, current_S)
        alpha_jumped = current_alpha + 1j * epsilon_cavity
        current_alpha = jnp.where(is_pulse, alpha_jumped, current_alpha)
        # ==========================================================

        S_hist_final = S_hist_updated.at[step_idx].set(current_S)
        alpha_hist_final = alpha_hist_updated.at[step_idx].set(current_alpha)

        return (S_hist_final, alpha_hist_final), None

    init_carry = (S_history, alpha_history)
    time_indices = jnp.arange(1, num_steps, dtype=jnp.int64)
    (final_S_history, final_alpha_history), _ = jax.lax.scan(scan_body, init_carry, time_indices)

    return final_S_history, final_alpha_history


# =====================================================================
# 3. GLOBAL BATCH COMPILER & WRAPPER
# =====================================================================

@jax.jit
def _accumulate_batch_sums(spin_ensemble: jax.Array, cavity_ensemble: jax.Array, j_val: float) -> dict:
    """Reduce a batch of trajectories into running ensemble sums.

    Implements the Z2 symmetry "folding" used in the superradiant phase: above the
    critical coupling each trajectory commits to one of two mirror-image wells
    (+psi_c, +J_x) or (-psi_c, -J_x), so the naive mean cancels to ~0. We fold every
    trajectory into the positive well before averaging the symmetry-breaking
    observables (J_x, J_y, psi_c), while symmetric observables (J_z) and quadratic
    observables (J_x^2, |J_x|, |psi_c|^2) are left untouched.

    Risk-1 fix: the fold sign is the sign of the *time-averaged* Re(psi_c) over the
    whole trajectory (a majority vote), not the sign of the single final snapshot.
    A late, isolated inter-well hop can no longer flip a trajectory's entire-history
    label.

    Parameters
    ----------
    spin_ensemble : float64 array, shape (batch_size, num_steps, 3)
        Spin trajectories for the batch.
    cavity_ensemble : complex128 array, shape (batch_size, num_steps)
        Cavity-amplitude trajectories for the batch.
    j_val : float
        Collective spin length j = N / 2 (intensive normalization).

    Returns
    -------
    dict of arrays
        Running sums over the batch axis:
        ``sum_jx``, ``sum_jy``  : folded intensive transverse magnetizations, shape (num_steps,)
        ``sum_jz``              : (unfolded) intensive longitudinal magnetization, shape (num_steps,)
        ``sum_jx_sq``           : sum of (J_x / j)^2, shape (num_steps,)
        ``sum_abs_jx``          : sum of |J_x / j|, shape (num_steps,)
        ``sum_psi``             : folded cavity amplitude, shape (num_steps,) complex
        ``sum_psi_sq``          : sum of |psi_c|^2, shape (num_steps,)
    """
    # Ensembles have shape: (batch_size, num_steps)
    jx_trajs = spin_ensemble[:, :, 0] / j_val
    jy_trajs = spin_ensemble[:, :, 1] / j_val
    jz_trajs = spin_ensemble[:, :, 2] / j_val

    # ==========================================================
    # --- THE FILTER: TRAJECTORY UNFOLDING ---
    # ==========================================================
    # 1. Decide each trajectory's well by a MAJORITY VOTE over its full history
    #    (time-averaged Re(psi_c)), not the single final snapshot. This is robust
    #    to an isolated late inter-well hop (Risk 1).
    mean_re_alpha = jnp.mean(jnp.real(cavity_ensemble), axis=1)
    signs = jnp.sign(mean_re_alpha)
    signs = jnp.where(signs == 0, 1.0, signs)  # Safety catch for exact 0
    signs = signs[:, None]  # Reshape to (batch_size, 1) for broadcasting

    # 2. Fold the broken-symmetry variables into the positive well
    folded_cavity = cavity_ensemble * signs
    folded_jx = jx_trajs * signs
    folded_jy = jy_trajs * signs
    # (Note: j_z is symmetric and points the same way in both wells, do not fold it!)
    # ==========================================================

    return {
        # Use the folded trajectories for the mean fields!
        "sum_jx": jnp.sum(folded_jx, axis=0),
        "sum_jy": jnp.sum(folded_jy, axis=0),
        "sum_jz": jnp.sum(jz_trajs, axis=0),

        # Squares and absolutes are immune to the negative sign, so keep them as is
        "sum_jx_sq": jnp.sum(jx_trajs**2, axis=0),
        "sum_abs_jx": jnp.sum(jnp.abs(jx_trajs), axis=0),

        "sum_psi": jnp.sum(folded_cavity, axis=0),
        "sum_psi_sq": jnp.sum(jnp.abs(cavity_ensemble)**2, axis=0),
    }

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling'])
def _compiled_master_processor(batched_keys: jax.Array, omega_0: float, B_field_safe: jax.Array, g: float,
                               n_photons_initial: complex, initial_direction: jax.Array, n_spins: int,
                               dt: float, num_steps: int, Sigma_R_t: jax.Array, cos_wt: jax.Array,
                               sin_wt: jax.Array, amp: jax.Array, use_noise: bool, use_sampling: bool,
                               pulse_idx: int, epsilon_spin: float, epsilon_cavity: float,
                               epsilon_spin_y: float = 0.0) -> dict:
    """vmap over trajectories and scan over batches, accumulating ensemble sums.

    Vectorizes :func:`solve_single_trajectory` across one batch of keys, reduces
    each batch with :func:`_accumulate_batch_sums`, and accumulates the running
    sums over all batches with a ``lax.scan``. Keeping only running sums (not the
    full trajectory tensor) bounds memory to O(num_steps^2) regardless of the
    total trajectory count.

    Parameters
    ----------
    batched_keys : PRNGKey array, shape (n_batches, batch_size, 2)
        Pre-batched RNG keys.
    omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins, dt, num_steps,
    Sigma_R_t, t_grid, w_pos, amp, use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity
        Forwarded to :func:`solve_single_trajectory` (see that function). ``n_spins``,
        ``num_steps``, ``use_noise`` and ``use_sampling`` are static (trigger
        recompilation when changed). ``n_photons_initial`` is used as ``alpha_shift``.

    Returns
    -------
    dict of arrays
        The summed statistics from :func:`_accumulate_batch_sums`, totalled across
        all trajectories.
    """
    j_val = n_spins / 2.0
    vmap_solver = jax.vmap(
        solve_single_trajectory,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry_stats, current_batch_keys):
        batch_S, batch_alpha = vmap_solver(
            current_batch_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction,
            n_spins, dt, num_steps, Sigma_R_t, cos_wt, sin_wt, amp,
            use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y
        )
        batch_sums = _accumulate_batch_sums(batch_S, batch_alpha, j_val)
        next_carry = {key: carry_stats[key] + batch_sums[key] for key in carry_stats}
        return next_carry, None

    init_stats = {
        "sum_jx": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_jy": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_jz": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_jx_sq": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_abs_jx": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_psi": jnp.zeros(num_steps, dtype=jnp.complex128),
        "sum_psi_sq": jnp.zeros(num_steps, dtype=jnp.float64),
    }

    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats


def run_dtwa(keys: jax.Array, t_grid: jax.Array, omega_0: float, alpha: float, omega_c: float,
             s: float, T: float, B_field: jax.Array, g: float, n_photons_initial: complex,
             initial_direction: jax.Array, batch_size: int = 1000, n_spins: int = 1,
             w_max: float = 40.0, N_w: int = 5000, use_noise: bool = True, use_sampling: bool = True,
             pulse_idx: int = -1, epsilon_spin: float = 0.0, epsilon_cavity: float = 0.0,
             mem_window: int = None, epsilon_spin_y: float = 0.0) -> dict:
    """Top-level driver: run the full DTWA ensemble and return ensemble averages.

    Builds the bath kernels, batches the supplied RNG keys, runs the compiled
    master processor, and normalizes the running sums into ensemble-averaged
    observables.

    Parameters
    ----------
    keys : PRNGKey array, shape (n_total, 2)
        One RNG key per trajectory. Trajectories beyond ``n_batches * batch_size``
        are dropped.
    t_grid : float64 array, shape (num_steps,)
        Uniform simulation time grid (dt and num_steps are inferred from it).
    omega_0 : float
        Bare cavity frequency.
    alpha : float
        Bath coupling strength of J(omega) (note: distinct from the cavity field).
    omega_c : float
        Bath cutoff frequency.
    s : float
        Bath ohmicity exponent.
    T : float
        Bath temperature (kB = 1).
    B_field : scalar, shape (3,), or shape (num_steps, 3)
        External field. A scalar is placed on the z-axis; a 3-vector is held
        constant in time; a full (num_steps, 3) array is used as-is.
    g : float
        Light-matter coupling.
    n_photons_initial : complex
        Mean initial cavity amplitude (passed as ``alpha_shift``).
    initial_direction : array, shape (3,)
        Initial spin direction.
    batch_size : int, optional
        Trajectories per compiled batch (memory/throughput trade-off).
    n_spins : int, optional
        Number of atoms N.
    w_max : float, optional
        Frequency-grid half-width for the bath kernels.
    N_w : int, optional
        Number of frequency grid points.
    use_noise : bool, optional
        Enable bath noise (TWA) vs. deterministic (mean-field) cavity.
    use_sampling : bool, optional
        Enable Wigner sampling of initial conditions.
    pulse_idx : int, optional
        Time index of the impulsive perturbation (default -1 = disabled).
    epsilon_spin : float, optional
        Spin-kick angle for linear response.
    epsilon_cavity : float, optional
        Cavity-kick magnitude for linear response.
    mem_window : int or None, optional
        [P1] Number of leading retarded-kernel taps to retain in the memory
        integral. ``None`` (default) keeps the full history (O(num_steps^2),
        exact). An integer L truncates to the last L steps (O(num_steps * L));
        choose L past where the printed kernel decay is negligible and confirm
        results are unchanged (convergence check).

    Returns
    -------
    dict of numpy.ndarray
        ``j_x``, ``j_y``, ``j_z``    : intensive magnetizations (folded for x, y), shape (num_steps,)
        ``rms_jx``                   : sqrt(<(J_x/j)^2>), the symmetry-robust order parameter
        ``abs_jx``                   : <|J_x/j|>
        ``mean_psi``                 : folded mean cavity amplitude (complex)
        ``abs_mean_psi``             : |mean_psi|
        ``mean_photon_number``       : <|psi_c|^2> - 1/2 (Wigner vacuum subtracted)
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]

    safe_w_max = w_max

    B_arr = jnp.asarray(B_field)
    if B_arr.ndim == 0:
        B_field_safe = jnp.zeros((num_steps, 3)).at[:, 2].set(B_arr)
    elif B_arr.shape == (3,):
        B_field_safe = jnp.tile(B_arr, (num_steps, 1))
    else:
        B_field_safe = jnp.reshape(B_arr, (num_steps, 3))

    Sigma_R_t, amp_full, w_grid, dw = compute_explicit_bath_kernels(
        num_steps, dt, omega_0, alpha, omega_c, s, T, safe_w_max, N_w)

    Sigma_R_t, t_grid_pre, cos_wt, sin_wt, amp = precompute_solver_arrays(
        num_steps, dt, Sigma_R_t, amp_full, dw, w_grid)

    # [P1] Memory-window truncation: keep only the leading L taps of the retarded
    # kernel (the part before it has decayed). None -> full O(N^2) history (exact).
    mag = np.abs(np.asarray(Sigma_R_t))
    mag = mag / (mag[0] + 1e-300)
    def _decay(tol):
        idx = int(np.argmax(mag < tol))
        return idx if idx > 0 else num_steps
    L = num_steps if mem_window is None else int(min(mem_window, num_steps))
    Sigma_R_t = Sigma_R_t[:L]
    print(f"Memory window L={L}/{num_steps}  (|Sigma_R| decays <1e-3 by step "
          f"{_decay(1e-3)}, <1e-4 by {_decay(1e-4)})")

    n_batches = n_total // batch_size
    batched_keys = keys[:n_batches * batch_size].reshape(n_batches, batch_size, -1)

    print(f"Executing {n_total} trajectories across {n_batches} compiled batches...")

    running_stats = _compiled_master_processor(
        batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction,
        n_spins, dt, num_steps, Sigma_R_t, cos_wt, sin_wt, amp,
        use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y
    )

    final_stats = {
        "j_x": running_stats["sum_jx"] / n_total,
        "j_y": running_stats["sum_jy"] / n_total,
        "j_z": running_stats["sum_jz"] / n_total,
        "rms_jx": jnp.sqrt(running_stats["sum_jx_sq"] / n_total),
        "abs_jx": running_stats["sum_abs_jx"] / n_total,
        "mean_psi": running_stats["sum_psi"] / n_total,
        "abs_mean_psi": jnp.abs(running_stats["sum_psi"] / n_total),
        "mean_photon_number": (running_stats["sum_psi_sq"] / n_total) - 0.5
    }

    final_stats_cpu = {key: np.array(value) for key, value in final_stats.items()}
    print("Simulation Complete!")

    return final_stats_cpu
