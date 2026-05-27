import numpy as np
import jax
import jax.numpy as jnp

from dtwa_non_integrated import (
    solve_single_trajectory,
    compute_explicit_bath_kernels,
    precompute_solver_matrices
)

# =====================================================================
# 1. TIME-DOMAIN CORRELATION ENGINE (O(N) Memory)
# =====================================================================

@jax.jit
def compute_exact_lag_correlation(x_batch):
    """
    Computes exact time-domain auto-correlation avoiding FFTs and O(N^2) memory.
    Works for both 2D (batched trajectories) and 1D (ensemble means) arrays.
    """
    x_2d = jnp.atleast_2d(x_batch)
    N = x_2d.shape[1]
    lags = jnp.arange(N)

    def scan_body(carry, tau):
        # Shift the array left by tau
        shifted = jnp.roll(x_2d, -tau, axis=-1)
        # Create a mask to ignore values that wrapped around the end
        valid_mask = jnp.arange(N) < (N - tau)
        
        # Multiply, mask, and sum over both time (axis=-1) and batch (if any)
        c_tau = jnp.sum(x_2d * shifted * valid_mask)
        return carry, c_tau

    # lax.scan executes sequentially, keeping memory strictly O(N)
    _, corr = jax.lax.scan(scan_body, None, lags)
    return corr

# =====================================================================
# 2. COMPILED MASTER ENGINE: UNIFIED PROCESSOR
# =====================================================================

@jax.jit(static_argnames=['pulse_idx'])
def _accumulate_batch_sums(spin_ensemble, cavity_ensemble, j_val, pulse_idx):
    sx = spin_ensemble[:, :, 0] / j_val
    r_alpha = jnp.real(cavity_ensemble)
    
    # 1. Extract only the stationary time-series
    sx_stat = sx[:, pulse_idx:]
    r_alpha_stat = r_alpha[:, pulse_idx:]
    
    # 2. Compute exact time-domain auto-correlation for the batch
    sum_corr_sx = compute_exact_lag_correlation(sx_stat)
    sum_corr_ra = compute_exact_lag_correlation(r_alpha_stat)
    
    return {
        "sum_sx": jnp.sum(sx, axis=0),
        "sum_r_alpha": jnp.sum(r_alpha, axis=0),
        "sum_corr_sx": sum_corr_sx,
        "sum_corr_ra": sum_corr_ra
    }

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling', 'pulse_idx'])
def _compiled_master_processor(batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
                               n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
                               use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity):
    
    j_val = n_spins / 2.0
    N_tau = num_steps - pulse_idx
    
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
        
        batch_sums = _accumulate_batch_sums(batch_S, batch_alpha, j_val, pulse_idx)
        next_carry = {key: carry_stats[key] + batch_sums[key] for key in carry_stats}
        return next_carry, None

    init_stats = {
        "sum_sx": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_r_alpha": jnp.zeros(num_steps, dtype=jnp.float64),
        "sum_corr_sx": jnp.zeros(N_tau, dtype=jnp.float64),
        "sum_corr_ra": jnp.zeros(N_tau, dtype=jnp.float64)
    }
    
    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats

# =====================================================================
# 3. SPECTRAL TRANSFORMS & CORRELATION EXTRACTION
# =====================================================================

@jax.jit
def compute_spectra(C_tau, chi_tau, dt, w_grid):
    N = len(C_tau)
    tau_grid = jnp.arange(N, dtype=jnp.float64) * dt
    exp_kernel = jnp.exp(-1j * w_grid[:, None] * tau_grid[None, :])

    # 1. TAPERING (Windowing)
    taper_start = int(0.5 * N)
    taper = jnp.ones(N, dtype=jnp.float64)
    cos_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * (jnp.arange(N - taper_start) / (N - taper_start))))
    taper = taper.at[taper_start:].set(cos_taper)

    # ==========================================================
    # 2. EXACT SIMPSON'S 1/3 COMPOSITE WEIGHTS
    # ==========================================================
    idx = jnp.arange(N)
    
    # Find the largest odd number <= N for the perfect Simpson's bulk
    N_simpson = N - (1 - (N % 2)) 
    
    # Standard Simpson's pattern: 1/3, 4/3, 2/3, 4/3 ... 1/3
    simpson_weights = jnp.where(idx < N_simpson, 
                        jnp.where(idx == 0, 1.0/3.0,
                        jnp.where(idx == N_simpson - 1, 1.0/3.0,
                        jnp.where(idx % 2 == 1, 4.0/3.0, 2.0/3.0))),
                        0.0)
    
    # Fallback: If N is even, apply Trapezoidal rule to the very last interval
    is_even = (N % 2 == 0)
    simpson_weights = simpson_weights + jnp.where(is_even & (idx == N - 2), 0.5, 0.0)
    simpson_weights = simpson_weights + jnp.where(is_even & (idx == N - 1), 0.5, 0.0)

    # Combine with the taper
    combined_weights = simpson_weights * taper
    # ==========================================================

    # 3. DC LEAKAGE REMOVAL
    tail_len = int(0.10 * N)
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
    N_tau = num_steps - pulse_idx

    B_base = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[:, 2].set(p.get('B_z', 0.0))

    Sigma_R_t, amp_full, w_grid_full, dw = compute_explicit_bath_kernels(
        num_steps, dt, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], w_max, N_w
    )
    
    Sigma_matrix_weighted, cos_wt, sin_wt, amp = precompute_solver_matrices(
        num_steps, dt, Sigma_R_t, amp_full, dw, w_grid_full
    )
    
    batched_keys = keys[:(n_total // p['batch_size']) * p['batch_size']].reshape(-1, p['batch_size'], 2)

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

    # --- CONSTRUCT STATIONARY CORRELATIONS VIA MEAN SUBTRACTION ---
    mean_sx = res_base["sum_sx"] / n_total
    mean_r_alpha = res_base["sum_r_alpha"] / n_total
    
    # Extract the stationary ensemble means
    mean_sx_stat = mean_sx[pulse_idx:]
    mean_ra_stat = mean_r_alpha[pulse_idx:]
    
    # Compute the disconnected part <x><x(tau)> exactly in the time domain
    mean_sx_corr = compute_exact_lag_correlation(mean_sx_stat)
    mean_ra_corr = compute_exact_lag_correlation(mean_ra_stat)

    # Normalization accounts for decreasing temporal overlap at larger lag times
    counts = N_tau - jnp.arange(N_tau)
    counts = jnp.where(counts > 0, counts, 1.0)
    
    # Assemble final stationary correlation: <x(t)x(t+tau)> - <x(t)><x(t+tau)>
    C_spin = ((res_base["sum_corr_sx"] / n_total) - mean_sx_corr) / counts
    C_cavity = (4.0 / j_val) * (((res_base["sum_corr_ra"] / n_total) - mean_ra_corr) / counts)

    # --- CONSTRUCT CAUSAL RESPONSES ---
    mean_sx_pert = res_pert_spin["sum_sx"] / n_total
    response_spin = np.array((mean_sx_pert - mean_sx)[pulse_idx:] / (epsilon * j_val))
    
    mean_r_alpha_pert = res_pert_cav["sum_r_alpha"] / n_total
    response_cavity = np.array(2.0 * (mean_r_alpha_pert - mean_r_alpha)[pulse_idx:] / (epsilon * j_val))

    # Slice grids (Using geometric space for resolving the 1/w pole)
    tau_grid = np.arange(N_tau) * dt
    w_min = 2.0 * jnp.pi / (N_tau * dt)
    w_grid = jnp.geomspace(w_min, w_max, N_w)

    print("Computing Fourier Transforms...")
    S_c_spin, S_chi_spin = compute_spectra(np.array(C_spin), response_spin, dt, w_grid)
    S_c_cavity, S_chi_cavity = compute_spectra(np.array(C_cavity), response_cavity, dt, w_grid)

    print("Correlations, Responses, and Spectra Complete!")
    
    return {
        "tau_grid": tau_grid,
        "w_grid": np.array(w_grid),
        "C_spin": np.array(C_spin),
        "C_cavity": np.array(C_cavity),
        "response_spin": response_spin,
        "response_cavity": response_cavity,
        "S_c_spin": np.array(S_c_spin),
        "S_chi_spin": np.array(S_chi_spin),
        "S_c_cavity": np.array(S_c_cavity),
        "S_chi_cavity": np.array(S_chi_cavity)
    }