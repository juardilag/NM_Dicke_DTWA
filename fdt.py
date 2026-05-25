import numpy as np
import jax
import jax.numpy as jnp

from dtwa_non_integrated import (
    solve_single_trajectory,
    compute_explicit_bath_kernels,
    precompute_solver_matrices
)

# =====================================================================
# 1. COMPILED MASTER ENGINE: UNIFIED PROCESSOR
# =====================================================================

@jax.jit
def _accumulate_batch_sums(spin_ensemble, cavity_ensemble, j_val):
    sx = spin_ensemble[:, :, 0] / j_val
    r_alpha = jnp.real(cavity_ensemble)
    
    return {
        "sum_sx": jnp.sum(sx, axis=0),
        "sum_r_alpha": jnp.sum(r_alpha, axis=0),
        "outer_sx": sx.T @ sx,
        "outer_r_alpha": r_alpha.T @ r_alpha
    }

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling'])
def _compiled_master_processor(batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
                               n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
                               use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity):
    
    j_val = n_spins / 2.0
    
    # Updated signature to 18 arguments matching the new solve_single_trajectory
    vmap_solver = jax.vmap(
        solve_single_trajectory, 
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry_stats, current_batch_keys):
        batch_S, batch_alpha = vmap_solver(
            current_batch_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
            n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
            use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity
        )
        
        batch_sums = _accumulate_batch_sums(batch_S, batch_alpha, j_val)
        next_carry = {key: carry_stats[key] + batch_sums[key] for key in carry_stats}
        return next_carry, None

    init_stats = {
        "sum_sx": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_r_alpha": jnp.zeros(num_steps, dtype=jnp.float64),
        "outer_sx": jnp.zeros((num_steps, num_steps), dtype=jnp.float64),
        "outer_r_alpha": jnp.zeros((num_steps, num_steps), dtype=jnp.float64)
    }
    
    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats


# =====================================================================
# 2. EXACT FOURIER TRANSFORMS & CORRELATION EXTRACTION
# =====================================================================

@jax.jit
def extract_stationary_correlation(C_matrix):
    n = C_matrix.shape[0]
    rows = jnp.arange(n, dtype=jnp.int64)
    def roll_row(i, row): return jnp.roll(row, -i)
    aligned_matrix = jax.vmap(roll_row)(rows, C_matrix)
    lags = jnp.arange(n, dtype=jnp.int64)
    counts = n - lags
    c_tau_sum = jnp.sum(aligned_matrix * (rows[:, None] + lags[None, :] < n), axis=0)
    return c_tau_sum / jnp.where(counts > 0, counts, 1.0)

@jax.jit
def compute_spectra(C_tau, chi_tau, dt, w_grid):
    N = len(C_tau)
    tau_grid = jnp.arange(N, dtype=jnp.float64) * dt
    exp_kernel = jnp.exp(-1j * w_grid[:, None] * tau_grid[None, :])

    # 1. TAPERING (Windowing)
    taper_start = int(0.85 * N)
    taper = jnp.ones(N, dtype=jnp.float64)
    cos_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * (jnp.arange(N - taper_start) / (N - taper_start))))
    taper = taper.at[taper_start:].set(cos_taper)

    # 2. INTEGRATION WEIGHTS
    weights = jnp.ones(N, dtype=jnp.float64).at[0].set(0.5).at[-1].set(0.5)
    combined_weights = weights * taper

    # 3. DC LEAKAGE REMOVAL
    tail_len = int(0.1 * N)
    C_tau = C_tau - jnp.mean(C_tau[-tail_len:])
    chi_tau = chi_tau - jnp.mean(chi_tau[-tail_len:])
    
    # 4. FOURIER TRANSFORMS
    S_w = 2.0 * jnp.real(jnp.dot(exp_kernel, C_tau * combined_weights) * dt)
    chi_w = jnp.dot(exp_kernel, chi_tau * combined_weights) * dt
    
    return S_w, -jnp.imag(chi_w)


def calculate_correlations_and_responses(keys, t_grid, p, t_pulse, epsilon=1e-5, w_max=20.0, N_w=5000):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    pulse_idx = int(np.searchsorted(t_grid, t_pulse))
    j_val = p['n_spins'] / 2.0

    # 1. Prepare Base Fields (ETD perturbation fields are removed)
    B_base = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[:, 2].set(p.get('B_z', 0.0))

    # 2. Precompute bath kernels ONCE
    Sigma_R_t, amp_full, w_grid_full, dw = compute_explicit_bath_kernels(
        num_steps, dt, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], w_max, N_w
    )
    
    Sigma_matrix_weighted, cos_wt, sin_wt, amp = precompute_solver_matrices(
        num_steps, dt, Sigma_R_t, amp_full, dw, w_grid_full
    )
    
    batched_keys = keys[:(n_total // p['batch_size']) * p['batch_size']].reshape(-1, p['batch_size'], 2)

    # --- SIMULATION PASSES ---
    print(f"Executing Pass 1/3: Base Ensemble ({n_total} trajectories)...")
    res_base = _compiled_master_processor(
        batched_keys, p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
        True, True, pulse_idx, 0.0, 0.0
    )

    print(f"Executing Pass 2/3: Spin-Perturbed Ensemble...")
    res_pert_spin = _compiled_master_processor(
        batched_keys, p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
        True, True, pulse_idx, epsilon, 0.0
    )

    print(f"Executing Pass 3/3: Cavity-Perturbed Ensemble...")
    res_pert_cav = _compiled_master_processor(
        batched_keys, p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
        True, True, pulse_idx, 0.0, epsilon
    )

    # --- CONSTRUCT STATIONARY CORRELATIONS ---
    mean_sx = res_base["sum_sx"] / n_total
    mean_r_alpha = res_base["sum_r_alpha"] / n_total
    
    C_spin_mat = (res_base["outer_sx"] / n_total) - jnp.outer(mean_sx, mean_sx)
    C_cavity_mat = 4.0*((res_base["outer_r_alpha"] / n_total) - jnp.outer(mean_r_alpha, mean_r_alpha)) / j_val 

    C_spin_mat_stat = C_spin_mat[pulse_idx:, pulse_idx:]
    C_cavity_mat_stat = C_cavity_mat[pulse_idx:, pulse_idx:]

    C_spin_full = extract_stationary_correlation(C_spin_mat_stat)
    C_cavity_full = extract_stationary_correlation(C_cavity_mat_stat)

    # --- CONSTRUCT CAUSAL RESPONSES ---
    mean_sx_pert = res_pert_spin["sum_sx"] / n_total
    response_spin_full = (mean_sx_pert - mean_sx) / (epsilon * j_val)
    
    mean_r_alpha_pert = res_pert_cav["sum_r_alpha"] / n_total
    response_cavity_full = 2.0 * (mean_r_alpha_pert - mean_r_alpha) / (epsilon * j_val)

    # Slice grids 
    start_idx = pulse_idx 
    N_tau = num_steps - start_idx
    tau_grid = np.arange(N_tau) * dt
    
    w_grid = jnp.linspace(0.0, 2.0, N_w)
    
    C_spin = np.array(C_spin_full[:N_tau])
    C_cavity = np.array(C_cavity_full[:N_tau])
    
    response_spin = np.array(response_spin_full[start_idx:])
    response_cavity = np.array(response_cavity_full[start_idx:])

    print("Computing Fourier Transforms...")
    S_c_spin, S_chi_spin = compute_spectra(C_spin, response_spin, dt, w_grid)
    S_c_cavity, S_chi_cavity = compute_spectra(C_cavity, response_cavity, dt, w_grid)

    print("Correlations, Responses, and Spectra Complete!")
    
    return {
        "tau_grid": tau_grid,
        "w_grid": np.array(w_grid),
        "C_spin": C_spin,
        "C_cavity": C_cavity,
        "response_spin": response_spin,
        "response_cavity": response_cavity,
        "S_c_spin": np.array(S_c_spin),
        "S_chi_spin": np.array(S_chi_spin),
        "S_c_cavity": np.array(S_c_cavity),
        "S_chi_cavity": np.array(S_chi_cavity)
    }