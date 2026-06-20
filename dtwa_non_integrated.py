"""
Discrete Truncated Wigner Approximation (DTWA) for the open Dicke model
with an *explicit*, non-integrated cavity mode AND an (optional) atomic bath.

This module implements coupled semiclassical equations of motion for a cavity
mode and a collective spin, each of which can be coupled to its own bath. Both
baths are independently configurable:

    * Markovian  vs  non-Markovian     (memoryless local damping vs memory kernel)
    * RWA        vs  full (no-RWA)      system-bath coupling
    * (spin bath only) transverse  vs  longitudinal coupling channel

The light-matter interaction itself can also be chosen full (Dicke) or RWA
(Tavis-Cummings) independently of the baths.

Equations of motion (schematic, units hbar = 1)
-----------------------------------------------
    Cavity (generalized Langevin equation):
        i d/dt psi(t) = omega_0 psi(t)  +  D_cav,bath[psi](t)
                        +  D_LM[S](t)    +  bath noise xi_cav(t)

    Spin (kinematic precession + optional dissipation):
        d/dt S(t) = B_eff(t) x S(t)  [ + Markovian spin dissipator ]
        B_eff(t)  = B  +  B_LM[psi](t)  +  B_spinbath[S](t)  +  eta_spin(t)

The cavity bath term ``D_cav,bath`` is either
    * non-Markovian:  -i * integral Sigma_R^cav(t - t') q(t') dt'
      where q = psi + psi* (full / position coupling, real kernel) or q = psi
      (RWA coupling, complex kernel); or
    * Markovian:      -kappa_cav * psi   (local Lindblad damping, the textbook
      quantum-optics limit with kappa = pi J(omega_0)).

The spin bath acts analogously through an effective field along the chosen
channel axis (S_x for 'transverse', S_z for 'longitudinal'); its Markovian limit
is a Bloch-type dissipator (transverse decay + longitudinal relaxation toward the
thermal S_z, or pure dephasing). See ``build_bath_kernels`` and
``non_markovian_coupled_heun_step`` for the precise expressions.

Bath spectral density (Ohmic family, both baths)
------------------------------------------------
    J(omega) = alpha * omega_c * (omega / omega_c)**s * exp(-omega / omega_c),  omega > 0,
with s < 1 sub-Ohmic, s = 1 Ohmic, s > 1 super-Ohmic.

Backward compatibility
----------------------
The defaults reproduce the previous behavior exactly: the cavity bath is
non-Markovian with full (position) coupling, there is NO spin bath
(``alpha_spin = 0``), and the light-matter interaction is full (Dicke).

Conventions
-----------
* Spins are stored as Cartesian vectors S = (S_x, S_y, S_z) with collective
  length j = N / 2 (intensive components are S / j).
* The cavity amplitude ``alpha`` is the complex coherent-state field psi.
* All real arrays are float64 and the cavity is complex128 (x64 enabled at
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

# Channel-axis lookup for the spin bath: 'transverse' couples to S_x, the
# quadrature analogous to the cavity X = psi + psi* and to the paper's atomic
# Holstein-Primakoff boson b ~ S_-; 'longitudinal' couples to S_z (pure dephasing).
_SPIN_AXIS = {"transverse": 0, "longitudinal": 2}

# =====================================================================
# 1. KERNELS & PRECOMPUTE ENGINE
# =====================================================================

def _ohmic_J(w_grid: jax.Array, alpha: float, omega_c: float, s: float) -> jax.Array:
    """Ohmic-family spectral density J(omega) on the positive-frequency support."""
    return jnp.where(
        w_grid > 1e-10,
        alpha * omega_c * (w_grid / omega_c) ** s * jnp.exp(-w_grid / omega_c),
        0.0,
    )


@jax.jit(static_argnames=['num_steps', 'N_w', 'markovian', 'rwa'])
def build_bath_kernels(num_steps: int, dt: float, omega_nat: float, alpha: float,
                       omega_c: float, s: float, T: float,
                       markovian: bool = False, rwa: bool = False,
                       w_max: float = 40.0, N_w: int = 5000) -> tuple:
    """Build the retarded memory kernel + noise spectrum for one bath.

    Generic over both baths (cavity and spin) and over the four coupling regimes
    selected by ``markovian`` and ``rwa``. The spectral density is the Ohmic
    family ``J(omega) = alpha omega_c (omega/omega_c)^s exp(-omega/omega_c)``.

    Regimes
    -------
    * **non-Markovian, full (rwa=False):** real Caldeira-Leggett friction kernel
          Sigma_R(t) = -2 theta(t) int_0^inf dw J(w) sin(w t),
      coupling to the position quadrature (psi + psi* / S_x); Im Sigma_R(w) =
      -pi J(w) is the dissipation and Re Sigma_R(w) the Lamb shift. Real colored
      noise with PSD 2 J(w) coth(w/2T).
    * **non-Markovian, RWA (rwa=True):** complex rotating-wave kernel
          Sigma_R(t) = -i theta(t) int_0^inf dw J(w) e^{-i w t},
      coupling to the lowering field (psi / S_-). Complex colored noise from the
      one-sided spectrum J(w) (coth(w/2T) + 1).
    * **Markovian (either rwa):** memoryless local damping with rate
          kappa = pi J(omega_nat)   (Fermi golden rule at the natural frequency),
      and white noise of symmetrized strength D = 2 kappa coth(omega_nat/2T). The
      kernel array is returned as zeros (unused; the solver applies -kappa
      locally). This is the standard quantum-optics Markovian limit and the one
      the paper uses for the cavity (P^K_ph = 2 i kappa).

    Parameters
    ----------
    num_steps : int (static)
        Length of the tabulated time grid t = [0, dt, ..., (num_steps-1) dt].
    dt : float
        Time-step size.
    omega_nat : float
        Natural frequency of the mode the bath dresses (omega_0 for the cavity,
        the precession frequency ~B_z for the spin). Sets the Markovian rate.
    alpha, omega_c, s, T : float
        Bath coupling strength, cutoff, ohmicity exponent and temperature.
    markovian : bool (static)
        Memoryless local damping if True; memory kernel if False.
    rwa : bool (static)
        Rotating-wave system-bath coupling if True; full (position) if False.
    w_max : float, optional
        Half-width of the symmetric frequency grid for the FT integrals.
    N_w : int (static), optional
        Number of frequency-grid points on [-w_max, w_max].

    Returns
    -------
    Sigma_R_t : complex128 array, shape (num_steps,)
        Retarded kernel (zeros in the Markovian case).
    amp_full : float64 array, shape (N_w,)
        Colored-noise amplitude on the *full* signed w grid (zeros if Markovian).
    kappa : float
        Markovian damping rate (0.0 if non-Markovian).
    white_D : float
        Symmetrized white-noise strength D = 2 kappa coth(omega_nat/2T)
        (0.0 if non-Markovian).
    w_grid : float64 array, shape (N_w,)
        Signed frequency grid [-w_max, w_max].
    dw : float
        Frequency spacing.
    """
    w_grid = jnp.linspace(-w_max, w_max, N_w, dtype=jnp.float64)
    dw = w_grid[1] - w_grid[0]
    J_w_pos = _ohmic_J(w_grid, alpha, omega_c, s)
    t_grid = jnp.arange(num_steps) * dt
    coth_w = 1.0 / jnp.tanh(w_grid / (2.0 * T + 1e-12))

    if markovian:
        # Local damping at the natural frequency: kappa = pi J(omega_nat).
        J_nat = alpha * omega_c * (jnp.abs(omega_nat) / omega_c) ** s * jnp.exp(-jnp.abs(omega_nat) / omega_c)
        J_nat = jnp.where(jnp.abs(omega_nat) > 1e-10, J_nat, 0.0)
        kappa = jnp.pi * J_nat
        coth_nat = 1.0 / jnp.tanh(jnp.abs(omega_nat) / (2.0 * T + 1e-12))
        white_D = 2.0 * kappa * coth_nat
        Sigma_R_t = jnp.zeros(num_steps, dtype=jnp.complex128)
        amp_full = jnp.zeros(N_w, dtype=jnp.float64)
        return Sigma_R_t, amp_full, kappa, white_D, w_grid, dw

    if rwa:
        # [RWA] complex rotating-wave kernel Sigma_R(t) = -i int_0^inf J(w) e^{-iwt} dw.
        # e^{-iwt} = cos - i sin, so K(t) = (-i cos@J - sin@J) dw.
        cos_mat = jnp.cos(t_grid[:, None] * w_grid[None, :])
        sin_mat = jnp.sin(t_grid[:, None] * w_grid[None, :])
        Sigma_R_t = ((-1j * jnp.dot(cos_mat, J_w_pos) - jnp.dot(sin_mat, J_w_pos)) * dw
                     ).astype(jnp.complex128)
        # One-sided (rotating-wave) noise: PSD J(w)(coth + 1) on w > 0.
        amp_full = jnp.sqrt(jnp.where(J_w_pos > 0.0, J_w_pos * (coth_w + 1.0) * dw, 0.0))
    else:
        # [Full / non-RWA] real Caldeira-Leggett friction kernel
        # Sigma_R(t) = -2 theta(t) int_0^inf J(w) sin(w t) dw, acting on the X quadrature.
        Sigma_R_t = (-2.0 * jnp.dot(jnp.sin(t_grid[:, None] * w_grid[None, :]), J_w_pos) * dw
                     ).astype(jnp.complex128)
        # FDT-consistent real symmetric noise, PSD 2 J(w) coth(w/2T) on the X quadrature.
        amp_full = jnp.sqrt(jnp.where(J_w_pos > 0.0, 2.0 * J_w_pos * coth_w * dw, 0.0))

    return Sigma_R_t, amp_full, 0.0, 0.0, w_grid, dw


# Backward-compatible alias: the previous code called this name with the old
# signature (cavity, non-Markovian, full). Keyword defaults preserve behavior.
def compute_explicit_bath_kernels(num_steps, dt, omega_0, alpha, omega_c, s, T,
                                  w_max=40.0, N_w=5000):
    """Deprecated thin wrapper around :func:`build_bath_kernels`.

    Reproduces the original cavity bath (non-Markovian, full/position coupling)
    and returns the original 4-tuple ``(Sigma_R_t, amp_full, w_grid, dw)``.
    """
    Sigma_R_t, amp_full, _kappa, _wD, w_grid, dw = build_bath_kernels(
        num_steps, dt, omega_0, alpha, omega_c, s, T,
        markovian=False, rwa=False, w_max=w_max, N_w=N_w)
    return Sigma_R_t, amp_full, w_grid, dw


@jax.jit(static_argnames=['num_steps'])
def precompute_solver_arrays(num_steps: int, dt: float, amp_full: jax.Array,
                             dw: float, w_grid: jax.Array) -> tuple:
    """Build the shared positive-frequency trig matrices for colored-noise synthesis.

    The colored-noise synthesizer needs only the positive-frequency branch. The
    cos/sin matrices depend solely on (t grid, w grid) and are therefore shared by
    *both* baths; each bath supplies its own positive-frequency amplitude
    (``amp_full[half:]``).

    Returns
    -------
    t_grid : float64 array, shape (num_steps,)
    cos_wt, sin_wt : float64 arrays, shape (num_steps, N_w // 2)
        Precomputed cos(t w) and sin(t w) on the positive-frequency grid.
    w_pos : float64 array, shape (N_w // 2,)
        Positive frequency grid (returned for convenience).
    """
    half_N = w_grid.shape[0] // 2
    w_pos = w_grid[half_N:]
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    wt = t_grid[:, None] * w_pos[None, :]
    return t_grid, jnp.cos(wt), jnp.sin(wt), w_pos


def positive_amp(amp_full: jax.Array) -> jax.Array:
    """Slice the positive-frequency half of a full-grid noise amplitude array."""
    return amp_full[amp_full.shape[0] // 2:]

# =====================================================================
# 2. HIGH-SPEED TRAJECTORY SOLVER (HEUN / ETDRK2)
# =====================================================================

def _windowed_memory(target_idx, hist, kernel, L, k_idx, dt, conjugate_quad):
    """Causal trapezoidal convolution sum  int Sigma_R(t-t') q(t') dt'.

    Evaluates the retarded memory integral over the last ``L`` history points
    (trapezoidal weights, strict causality). ``hist`` is the field history;
    ``conjugate_quad`` selects the X-quadrature (hist + conj(hist), full/position
    coupling) vs the bare field (hist, RWA coupling).
    """
    start = jnp.maximum(target_idx - L + 1, 0)
    block = jax.lax.dynamic_slice(hist, (start,), (L,))
    block_j = start + k_idx
    lag = target_idx - block_j
    valid = lag >= 0
    safe_lag = jnp.where(valid, lag, 0)
    sigma_vals = jnp.where(valid, kernel[safe_lag], 0.0 + 0.0j)

    weights = jnp.where(valid, 1.0, 0.0)
    weights = jnp.where(block_j == 0, 0.5, weights)
    weights = jnp.where(block_j == target_idx, 0.5, weights)
    weights = jnp.where(target_idx == 0, 0.0, weights)

    q = block + jnp.conj(block) if conjugate_quad else block
    return jnp.dot(sigma_vals * weights * dt, q)


def non_markovian_coupled_heun_step(
        S_history, alpha_history, step_idx,
        cav_force_curr, cav_force_next, eta_spin_curr, eta_spin_next,
        Sigma_R_cav, Sigma_R_spin,
        B_field_val, coupling_strength, omega_0, dt, j_val,
        kappa_cav, kappa_spin, Szeq,
        cav_markovian, cav_rwa, rwa_interaction,
        spin_bath_on, spin_markovian, spin_axis):
    """Advance the coupled spin+cavity state by one Heun predictor-corrector step.

    The spin is rotated rigidly about the instantaneous effective field (Rodrigues
    rotation, norm-preserving), then -- if the spin bath is Markovian -- a Bloch
    dissipator is applied. The cavity is advanced with an integrating-factor
    (ETDRK2) predictor-corrector: the bare oscillation -i omega_0 psi is exact,
    while memory/drive/noise are handled to 2nd order. All bath behavior is set by
    the static flags below.

    Light-matter coupling (``rwa_interaction``)
    -------------------------------------------
    * full (Dicke):  cavity drive -i 2c S_x;  B_eff_x += 4c Re(psi)
    * RWA  (Tavis-Cummings):  cavity drive -i c (S_x - i S_y);
                              B_eff_x += 2c Re(psi),  B_eff_y += -2c Im(psi)
    with c = g / sqrt(N).

    Cavity bath
    -----------
    * Markovian:      d/dt psi += -kappa_cav psi  + cav_force
    * non-Markovian:  d/dt psi += -i (memory)     + cav_force
      memory on (psi + psi*) if full coupling, on psi if RWA coupling.

    Spin bath (only if ``spin_bath_on``)
    ------------------------------------
    * non-Markovian:  B_eff[axis] += Re(memory of S[axis]/j) + eta_spin
    * Markovian:      Bloch dissipator applied after the rotation (see below).
    """
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    alpha_curr = alpha_history[curr_idx]

    L_cav = Sigma_R_cav.shape[0]
    k_cav = jnp.arange(L_cav)
    L_spin = Sigma_R_spin.shape[0]
    k_spin = jnp.arange(L_spin)

    # --- [A2] Integrating-factor (ETDRK2) coefficients for dpsi/dt = -i w0 psi + N.
    z = -1j * omega_0 * dt
    E = jnp.exp(z)
    _small = jnp.abs(z) < 1e-8
    phi1 = jnp.where(_small, 1.0 + z / 2.0, (E - 1.0) / z)
    phi2 = jnp.where(_small, 0.5 + z / 6.0, (E - 1.0 - z) / (z * z))
    b1 = dt * (phi1 - phi2)
    b2 = dt * phi2

    def cavity_bath_drive(alpha_val, hist, idx):
        # -kappa psi (Markovian) or -i memory (non-Markovian).
        if cav_markovian:
            return -kappa_cav * alpha_val
        mem = _windowed_memory(idx, hist, Sigma_R_cav, L_cav, k_cav, dt,
                               conjugate_quad=(not cav_rwa))
        return -1j * mem

    def light_matter_cavity_drive(Sx, Sy):
        if rwa_interaction:
            return -1j * coupling_strength * (Sx - 1j * Sy)
        return -1j * 2.0 * coupling_strength * Sx

    def spin_bath_field(idx):
        # Effective field along the channel axis from the (non-Markovian) spin bath.
        if (not spin_bath_on) or spin_markovian:
            return 0.0
        axis_hist = S_history[:, spin_axis] / j_val
        mem = _windowed_memory(idx, axis_hist, Sigma_R_spin, L_spin, k_spin, dt,
                               conjugate_quad=False)
        return jnp.real(mem)

    def build_B_eff(alpha_val, spin_field_axis, eta):
        # Base external field + light-matter + spin-bath field along the channel axis.
        bx = B_field_val[0]
        by = B_field_val[1]
        bz = B_field_val[2]
        if rwa_interaction:
            bx = bx + 2.0 * coupling_strength * jnp.real(alpha_val)
            by = by - 2.0 * coupling_strength * jnp.imag(alpha_val)
        else:
            bx = bx + 4.0 * coupling_strength * jnp.real(alpha_val)
        comp = [bx, by, bz]
        comp[spin_axis] = comp[spin_axis] + spin_field_axis + eta
        return jnp.array(comp, dtype=jnp.float64)

    def rotate(S, B_eff):
        b_mag = jnp.linalg.norm(B_eff) + 1e-16
        axis = B_eff / b_mag
        angle = b_mag * dt
        return (S * jnp.cos(angle)
                + jnp.cross(axis, S) * jnp.sin(angle)
                + axis * jnp.dot(axis, S) * (1.0 - jnp.cos(angle)))

    # --- 1. PREDICTOR ---
    spin_field_p = spin_bath_field(curr_idx)
    B_eff_p = build_B_eff(alpha_curr, spin_field_p, eta_spin_curr)
    S_pred = rotate(S_curr, B_eff_p)

    drive_p = (cavity_bath_drive(alpha_curr, alpha_history, curr_idx)
               + light_matter_cavity_drive(S_curr[0], S_curr[1])
               + cav_force_curr)
    alpha_pred = E * alpha_curr + dt * phi1 * drive_p
    alpha_history_pred = alpha_history.at[step_idx].set(alpha_pred)

    # --- 2. CORRECTOR ---
    B_eff_c = build_B_eff(alpha_pred, spin_field_p, eta_spin_next)
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    S_next = rotate(S_curr, B_eff_avg)

    drive_c = (cavity_bath_drive(alpha_pred, alpha_history_pred, step_idx)
               + light_matter_cavity_drive(S_pred[0], S_pred[1])
               + cav_force_next)
    alpha_next = E * alpha_curr + b1 * drive_p + b2 * drive_c

    # --- 3. Markovian spin dissipator (Bloch), applied after the rotation ---
    if spin_bath_on and spin_markovian:
        g_perp = kappa_spin
        Sx, Sy, Sz = S_next[0], S_next[1], S_next[2]
        if spin_axis == _SPIN_AXIS["transverse"]:
            # Transverse decay of S_x, S_y + longitudinal relaxation toward Szeq.
            Sx = Sx - g_perp * Sx * dt + eta_spin_next
            Sy = Sy - g_perp * Sy * dt + eta_spin_curr
            Sz = Sz - g_perp * (Sz - Szeq) * dt
        else:
            # Longitudinal channel (S_z coupling) = pure dephasing of the transverse spin.
            Sx = Sx - g_perp * Sx * dt + eta_spin_next
            Sy = Sy - g_perp * Sy * dt + eta_spin_curr
        S_next = jnp.array([Sx, Sy, Sz], dtype=jnp.float64)

    return S_history.at[step_idx].set(S_next), alpha_history.at[step_idx].set(alpha_next)


def solve_single_trajectory(
        key, omega_0, B_field_base, g, alpha_shift, initial_direction, n_spins,
        dt, num_steps,
        Sigma_R_cav, Sigma_R_spin, cos_wt, sin_wt, amp_cav, amp_spin,
        kappa_cav, kappa_spin, white_D_cav, white_D_spin, Szeq,
        use_noise, use_sampling,
        pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y,
        cav_markovian, cav_rwa, rwa_interaction,
        spin_bath_on, spin_markovian, spin_axis):
    """Integrate one coupled spin+cavity DTWA trajectory with configurable baths.

    Samples the initial spin (discrete Wigner) and cavity amplitude (Gaussian
    Wigner), synthesizes the cavity and (optional) spin bath noise according to
    the bath flags, then time-steps the coupled equations. An optional impulsive
    perturbation ("pulse") at a single time index enables linear-response
    measurements.

    The bath behavior is controlled by the static flags ``cav_markovian``,
    ``cav_rwa``, ``rwa_interaction``, ``spin_bath_on``, ``spin_markovian`` and the
    integer ``spin_axis`` (0 = transverse / S_x, 2 = longitudinal / S_z).
    """
    k_spin_samp, k_alpha_samp, k_cav_noise, k_spin_noise = jax.random.split(key, 4)
    coupling_strength = g / jnp.sqrt(n_spins)
    j_val = n_spins / 2.0

    s0_sampled = discrete_spin_sampling_factorized(k_spin_samp, initial_direction, n_spins) / 2.0
    s0_mean = (initial_direction * n_spins) / 2.0
    s0 = jnp.where(use_sampling, s0_sampled, s0_mean)

    alpha0_mean = jnp.array(alpha_shift, dtype=jnp.complex128)
    alpha0_sampled = cavity_wigner_sampling(k_alpha_samp, alpha0_mean)
    alpha0 = jnp.where(use_sampling, alpha0_sampled, alpha0_mean)

    S_history = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[0].set(s0)
    alpha_history = jnp.zeros((num_steps,), dtype=jnp.complex128).at[0].set(alpha0)

    # ---------------- Cavity bath force on d/dt psi ----------------
    if cav_markovian:
        kc1, kc2 = jax.random.split(k_cav_noise)
        g1 = jax.random.normal(kc1, (num_steps,), dtype=jnp.float64)
        g2 = jax.random.normal(kc2, (num_steps,), dtype=jnp.float64)
        # Complex white force, symmetrized strength white_D_cav.
        cav_force = jnp.sqrt(white_D_cav / dt) * (g1 + 1j * g2) / jnp.sqrt(2.0)
    else:
        half = amp_cav.shape[0]
        kc1, kc2 = jax.random.split(k_cav_noise)
        u = jax.random.normal(kc1, (half,), dtype=jnp.float64) * amp_cav
        v = jax.random.normal(kc2, (half,), dtype=jnp.float64) * amp_cav
        xi_re = cos_wt @ u + sin_wt @ v
        if cav_rwa:
            # Complex analytic-signal noise (positive frequencies only).
            xi_im = cos_wt @ v - sin_wt @ u
            xi = xi_re + 1j * xi_im
        else:
            xi = xi_re.astype(jnp.complex128)
        # Bath force enters d/dt psi as +1j*xi (matches the original convention).
        cav_force = 1j * xi
    cav_force = jnp.where(use_noise, cav_force, 0.0 + 0.0j)

    # ---------------- Spin bath noise (field for non-Markovian, kicks for Markovian) -----
    if spin_bath_on:
        if spin_markovian:
            ks1, ks2 = jax.random.split(k_spin_noise)
            std = jnp.sqrt(white_D_spin * j_val * dt)
            eta_a = jax.random.normal(ks1, (num_steps,), dtype=jnp.float64) * std
            eta_b = jax.random.normal(ks2, (num_steps,), dtype=jnp.float64) * std
        else:
            half = amp_spin.shape[0]
            ks1, ks2 = jax.random.split(k_spin_noise)
            us = jax.random.normal(ks1, (half,), dtype=jnp.float64) * amp_spin
            vs = jax.random.normal(ks2, (half,), dtype=jnp.float64) * amp_spin
            eta_a = cos_wt @ us + sin_wt @ vs
            eta_b = eta_a
        eta_a = jnp.where(use_noise, eta_a, 0.0)
        eta_b = jnp.where(use_noise, eta_b, 0.0)
    else:
        eta_a = jnp.zeros(num_steps, dtype=jnp.float64)
        eta_b = jnp.zeros(num_steps, dtype=jnp.float64)

    def scan_body(carry, step_idx):
        S_hist, alpha_hist = carry
        curr_idx = step_idx - 1

        B_val_step = jax.lax.dynamic_index_in_dim(B_field_base, curr_idx, axis=0, keepdims=False)

        S_hist_updated, alpha_hist_updated = non_markovian_coupled_heun_step(
            S_hist, alpha_hist, step_idx,
            cav_force[curr_idx], cav_force[step_idx], eta_a[curr_idx], eta_b[step_idx],
            Sigma_R_cav, Sigma_R_spin,
            B_val_step, coupling_strength, omega_0, dt, j_val,
            kappa_cav, kappa_spin, Szeq,
            cav_markovian, cav_rwa, rwa_interaction,
            spin_bath_on, spin_markovian, spin_axis,
        )

        current_S = S_hist_updated[step_idx]
        current_alpha = alpha_hist_updated[step_idx]

        # Impulsive linear-response kicks (exact analytical rotations / displacement).
        is_pulse = (step_idx == pulse_idx)
        cx, sx = jnp.cos(epsilon_spin), jnp.sin(epsilon_spin)
        Sy1 = current_S[1] * cx - current_S[2] * sx
        Sz1 = current_S[2] * cx + current_S[1] * sx
        cy, sy = jnp.cos(epsilon_spin_y), jnp.sin(epsilon_spin_y)
        Sx2 = current_S[0] * cy + Sz1 * sy
        Sz2 = -current_S[0] * sy + Sz1 * cy
        S_jumped = jnp.array([Sx2, Sy1, Sz2])

        current_S = jnp.where(is_pulse, S_jumped, current_S)
        current_alpha = jnp.where(is_pulse, current_alpha + 1j * epsilon_cavity, current_alpha)

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
    """Reduce a batch of trajectories into running ensemble sums (with Z2 folding).

    See the module docstring for the folding rationale. Broken-symmetry variables
    (J_x, J_y, psi) are folded into the positive well by the majority-vote sign of
    the time-averaged Re(psi); symmetric (J_z) and quadratic observables are left
    untouched.
    """
    jx_trajs = spin_ensemble[:, :, 0] / j_val
    jy_trajs = spin_ensemble[:, :, 1] / j_val
    jz_trajs = spin_ensemble[:, :, 2] / j_val

    mean_re_alpha = jnp.mean(jnp.real(cavity_ensemble), axis=1)
    signs = jnp.sign(mean_re_alpha)
    signs = jnp.where(signs == 0, 1.0, signs)[:, None]

    folded_cavity = cavity_ensemble * signs
    folded_jx = jx_trajs * signs
    folded_jy = jy_trajs * signs

    return {
        "sum_jx": jnp.sum(folded_jx, axis=0),
        "sum_jy": jnp.sum(folded_jy, axis=0),
        "sum_jz": jnp.sum(jz_trajs, axis=0),
        "sum_jx_sq": jnp.sum(jx_trajs ** 2, axis=0),
        "sum_abs_jx": jnp.sum(jnp.abs(jx_trajs), axis=0),
        "sum_psi": jnp.sum(folded_cavity, axis=0),
        "sum_psi_sq": jnp.sum(jnp.abs(cavity_ensemble) ** 2, axis=0),
    }


@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling',
                          'cav_markovian', 'cav_rwa', 'rwa_interaction',
                          'spin_bath_on', 'spin_markovian', 'spin_axis'])
def _compiled_master_processor(
        batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins,
        dt, num_steps,
        Sigma_R_cav, Sigma_R_spin, cos_wt, sin_wt, amp_cav, amp_spin,
        kappa_cav, kappa_spin, white_D_cav, white_D_spin, Szeq,
        use_noise, use_sampling,
        pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y,
        cav_markovian, cav_rwa, rwa_interaction,
        spin_bath_on, spin_markovian, spin_axis):
    """vmap over trajectories and scan over batches, accumulating ensemble sums.

    The static bath flags are closed over so the per-trajectory solver specializes
    to the selected regime; vmap maps only over the RNG keys.
    """
    j_val = n_spins / 2.0

    def solve_one(key):
        return solve_single_trajectory(
            key, omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins,
            dt, num_steps,
            Sigma_R_cav, Sigma_R_spin, cos_wt, sin_wt, amp_cav, amp_spin,
            kappa_cav, kappa_spin, white_D_cav, white_D_spin, Szeq,
            use_noise, use_sampling,
            pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y,
            cav_markovian, cav_rwa, rwa_interaction,
            spin_bath_on, spin_markovian, spin_axis)

    vmap_solver = jax.vmap(solve_one)

    def master_scan_body(carry_stats, current_batch_keys):
        batch_S, batch_alpha = vmap_solver(current_batch_keys)
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


def _resolve_bath_config(omega_0, alpha, omega_c, s, T, B_z, n_spins, dt, num_steps,
                         w_max, N_w,
                         cavity_bath_markovian, rwa_cavity_bath,
                         alpha_spin, omega_c_spin, s_spin, T_spin,
                         spin_bath_markovian, rwa_spin_bath, spin_bath_channel):
    """Build both bath kernels + shared trig matrices and return solver-ready arrays.

    Centralizes the kernel construction shared by :func:`run_dtwa` and the FDT
    driver. Spin-bath parameters default to the cavity-bath values when not given;
    ``alpha_spin = 0`` disables the spin bath entirely (``spin_bath_on = False``).
    """
    omega_c_spin = omega_c if omega_c_spin is None else omega_c_spin
    s_spin = s if s_spin is None else s_spin
    T_spin = T if T_spin is None else T_spin
    spin_bath_on = bool(alpha_spin and alpha_spin > 0.0)
    spin_axis = _SPIN_AXIS[spin_bath_channel]

    # Cavity bath (natural frequency = omega_0).
    Sig_cav, amp_cav_full, kappa_cav, white_D_cav, w_grid, dw = build_bath_kernels(
        num_steps, dt, omega_0, alpha, omega_c, s, T,
        markovian=cavity_bath_markovian, rwa=rwa_cavity_bath, w_max=w_max, N_w=N_w)

    # Spin bath (natural frequency ~ B_z precession). Built even when off so the
    # solver always has correctly-shaped arrays; with alpha_spin=0 it is inert.
    Sig_spin, amp_spin_full, kappa_spin, white_D_spin, _wg2, _dw2 = build_bath_kernels(
        num_steps, dt, B_z, (alpha_spin or 0.0), omega_c_spin, s_spin, T_spin,
        markovian=spin_bath_markovian, rwa=rwa_spin_bath, w_max=w_max, N_w=N_w)

    t_grid_pre, cos_wt, sin_wt, w_pos = precompute_solver_arrays(num_steps, dt, amp_cav_full, dw, w_grid)
    amp_cav = positive_amp(amp_cav_full)
    amp_spin = positive_amp(amp_spin_full)

    # Thermal S_z target for the Markovian spin relaxation.
    Szeq = -(n_spins / 2.0) * np.tanh(abs(B_z) / (2.0 * T_spin + 1e-12))

    return {
        "Sigma_R_cav": Sig_cav, "Sigma_R_spin": Sig_spin,
        "cos_wt": cos_wt, "sin_wt": sin_wt,
        "amp_cav": amp_cav, "amp_spin": amp_spin,
        "kappa_cav": float(kappa_cav), "kappa_spin": float(kappa_spin),
        "white_D_cav": float(white_D_cav), "white_D_spin": float(white_D_spin),
        "Szeq": float(Szeq),
        "spin_bath_on": spin_bath_on, "spin_axis": spin_axis,
    }


def run_dtwa(keys: jax.Array, t_grid: jax.Array, omega_0: float, alpha: float, omega_c: float,
             s: float, T: float, B_field: jax.Array, g: float, n_photons_initial: complex,
             initial_direction: jax.Array, batch_size: int = 1000, n_spins: int = 1,
             w_max: float = 40.0, N_w: int = 5000, use_noise: bool = True, use_sampling: bool = True,
             pulse_idx: int = -1, epsilon_spin: float = 0.0, epsilon_cavity: float = 0.0,
             mem_window: int = None, epsilon_spin_y: float = 0.0,
             # --- bath / interaction selection (new) ---
             cavity_bath_markovian: bool = False, rwa_cavity_bath: bool = False,
             alpha_spin: float = 0.0, omega_c_spin: float = None, s_spin: float = None,
             T_spin: float = None, spin_bath_markovian: bool = False,
             rwa_spin_bath: bool = False, spin_bath_channel: str = "transverse",
             rwa_interaction: bool = False, B_z: float = None) -> dict:
    """Top-level driver: run the full DTWA ensemble and return ensemble averages.

    New selection arguments (all default to the previous behavior)
    --------------------------------------------------------------
    cavity_bath_markovian : bool
        Cavity bath memoryless (Lindblad kappa) if True; memory kernel if False.
    rwa_cavity_bath : bool
        Rotating-wave cavity-bath coupling if True; full/position if False.
    alpha_spin : float
        Spin-bath coupling strength. ``0`` (default) disables the spin bath.
    omega_c_spin, s_spin, T_spin : float or None
        Spin-bath cutoff, ohmicity and temperature (default to the cavity values).
    spin_bath_markovian : bool
        Spin bath Bloch dissipator if True; memory field if False.
    rwa_spin_bath : bool
        Rotating-wave spin-bath kernel if True; full/position if False. (In the
        non-Markovian branch this selects the kernel construction; the field is
        applied along the channel axis -- see :func:`build_bath_kernels`.)
    spin_bath_channel : {'transverse', 'longitudinal'}
        Couple the spin bath to S_x (transverse, default; mirrors the paper's
        atomic mode) or S_z (longitudinal, pure dephasing).
    rwa_interaction : bool
        Light-matter coupling RWA (Tavis-Cummings) if True; full (Dicke) if False.
    B_z : float or None
        Spin precession frequency used to set the Markovian spin rate / S_z target.
        Defaults to the z-component of ``B_field`` (or 0 if not inferable).

    Returns
    -------
    dict of numpy.ndarray
        ``j_x``, ``j_y``, ``j_z``, ``rms_jx``, ``abs_jx``, ``mean_psi``,
        ``abs_mean_psi``, ``mean_photon_number`` (see the original docstring).
    """
    dt = float(t_grid[1] - t_grid[0])
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]

    B_arr = jnp.asarray(B_field)
    if B_arr.ndim == 0:
        B_field_safe = jnp.zeros((num_steps, 3)).at[:, 2].set(B_arr)
        B_z_val = float(B_arr) if B_z is None else B_z
    elif B_arr.shape == (3,):
        B_field_safe = jnp.tile(B_arr, (num_steps, 1))
        B_z_val = float(B_arr[2]) if B_z is None else B_z
    else:
        B_field_safe = jnp.reshape(B_arr, (num_steps, 3))
        B_z_val = float(B_arr.reshape(num_steps, 3)[0, 2]) if B_z is None else B_z

    cfg = _resolve_bath_config(
        omega_0, alpha, omega_c, s, T, B_z_val, n_spins, dt, num_steps, w_max, N_w,
        cavity_bath_markovian, rwa_cavity_bath,
        alpha_spin, omega_c_spin, s_spin, T_spin,
        spin_bath_markovian, rwa_spin_bath, spin_bath_channel)

    # [P1] Memory-window truncation of the cavity kernel (None -> full, exact).
    Sig_cav = cfg["Sigma_R_cav"]
    Sig_spin = cfg["Sigma_R_spin"]
    if not cavity_bath_markovian:
        mag = np.abs(np.asarray(Sig_cav)); mag = mag / (mag[0] + 1e-300)
        def _decay(tol):
            idx = int(np.argmax(mag < tol)); return idx if idx > 0 else num_steps
        L = num_steps if mem_window is None else int(min(mem_window, num_steps))
        Sig_cav = Sig_cav[:L]
        print(f"Memory window L={L}/{num_steps}  (|Sigma_R^cav| decays <1e-3 by step "
              f"{_decay(1e-3)}, <1e-4 by {_decay(1e-4)})")
    else:
        Sig_cav = Sig_cav[:1]  # unused; keep a length-1 array for shape sanity
        print(f"Cavity bath: MARKOVIAN (kappa={cfg['kappa_cav']:.4g})")
    # Spin kernel uses the same window length when active (memory branch).
    if cfg["spin_bath_on"] and not spin_bath_markovian:
        L_s = Sig_cav.shape[0]
        Sig_spin = Sig_spin[:max(L_s, 1)]
    else:
        Sig_spin = Sig_spin[:1]

    n_batches = n_total // batch_size
    batched_keys = keys[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    print(f"Executing {n_total} trajectories across {n_batches} compiled batches...")

    running_stats = _compiled_master_processor(
        batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins,
        dt, num_steps,
        Sig_cav, Sig_spin, cfg["cos_wt"], cfg["sin_wt"], cfg["amp_cav"], cfg["amp_spin"],
        cfg["kappa_cav"], cfg["kappa_spin"], cfg["white_D_cav"], cfg["white_D_spin"], cfg["Szeq"],
        use_noise, use_sampling,
        pulse_idx, epsilon_spin, epsilon_cavity, epsilon_spin_y,
        cavity_bath_markovian, rwa_cavity_bath, rwa_interaction,
        cfg["spin_bath_on"], spin_bath_markovian, cfg["spin_axis"])

    final_stats = {
        "j_x": running_stats["sum_jx"] / n_total,
        "j_y": running_stats["sum_jy"] / n_total,
        "j_z": running_stats["sum_jz"] / n_total,
        "rms_jx": jnp.sqrt(running_stats["sum_jx_sq"] / n_total),
        "abs_jx": running_stats["sum_abs_jx"] / n_total,
        "mean_psi": running_stats["sum_psi"] / n_total,
        "abs_mean_psi": jnp.abs(running_stats["sum_psi"] / n_total),
        "mean_photon_number": (running_stats["sum_psi_sq"] / n_total) - 0.5,
    }
    final_stats_cpu = {key: np.array(value) for key, value in final_stats.items()}
    print("Simulation Complete!")
    return final_stats_cpu
