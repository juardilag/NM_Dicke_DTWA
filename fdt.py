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
    sx = spin_ensemble[:, :, 0] / j_val
    r_alpha = jnp.real(cavity_ensemble)

    sx_stat = sx[:, pulse_idx:]
    r_alpha_stat = r_alpha[:, pulse_idx:]

    sum_corr_sx = compute_exact_lag_correlation(sx_stat)
    sum_corr_ra = compute_exact_lag_correlation(r_alpha_stat)

    return {
        "sum_sx": jnp.sum(sx, axis=0),
        "sum_r_alpha": jnp.sum(r_alpha, axis=0),
        "sum_corr_sx": sum_corr_sx,
        "sum_corr_ra": sum_corr_ra
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
        "sum_sx": jnp.zeros(num_steps),
        "sum_r_alpha": jnp.zeros(num_steps),
        "sum_corr_sx": jnp.zeros(N_tau),
        "sum_corr_ra": jnp.zeros(N_tau)
    }

    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats

# =====================================================================
# 3. SPECTRAL TRANSFORMS
# =====================================================================

@jax.jit
def compute_spectra(C_tau, chi_tau, dt, w_grid):
    N = len(C_tau)
    tau_grid = jnp.arange(N) * dt
    exp_kernel = jnp.exp(-1j * w_grid[:, None] * tau_grid[None, :])

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

    # === Correlations ===
    mean_sx = res_base["sum_sx"] / n_total
    mean_ra = res_base["sum_r_alpha"] / n_total

    mean_sx_stat = mean_sx[pulse_idx:]
    mean_ra_stat = mean_ra[pulse_idx:]

    mean_sx_corr = compute_exact_lag_correlation(mean_sx_stat)
    mean_ra_corr = compute_exact_lag_correlation(mean_ra_stat)

    counts = N_tau - jnp.arange(N_tau)
    counts = jnp.where(counts > 0, counts, 1.0)

    C_spin = ((res_base["sum_corr_sx"] / n_total) - mean_sx_corr) / counts
    C_cavity = (4.0 / j_val) * ((res_base["sum_corr_ra"] / n_total) - mean_ra_corr) / counts

    # === Responses ===
    response_spin = ((res_pert_spin["sum_sx"] - res_base["sum_sx"]) / n_total)[pulse_idx:] / (epsilon * j_val)
    response_cavity = 2.0 * ((res_pert_cav["sum_r_alpha"] - res_base["sum_r_alpha"]) / n_total)[pulse_idx:] / (epsilon * j_val)

    # === Spectra ===
    tau_grid = np.arange(N_tau) * dt
    w_min = 2.0 * jnp.pi / (N_tau * dt)
    w_grid = jnp.geomspace(w_min, w_max, N_w)

    print("Fourier transforms...")
    S_c_spin, S_chi_spin = compute_spectra(np.array(C_spin), np.array(response_spin), dt, w_grid)
    S_c_cavity, S_chi_cavity = compute_spectra(np.array(C_cavity), np.array(response_cavity), dt, w_grid)

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