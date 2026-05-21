import numpy as np
import jax
import jax.numpy as jnp

from dtwa_non_integrated import (
    solve_single_trajectory,
    compute_explicit_bath_kernels,
    precompute_solver_matrices
)

# =====================================================================
# 1. COMPILED MASTER ENGINE: CORRELATIONS
# =====================================================================

@jax.jit
def _accumulate_correlation_components(spin_ensemble, cavity_ensemble, j_val):
    sx = spin_ensemble[:, :, 0] / j_val
    r_alpha = jnp.real(cavity_ensemble)
    
    sum_sx = jnp.sum(sx, axis=0)
    sum_r_alpha = jnp.sum(r_alpha, axis=0)
    
    outer_sx = sx.T @ sx
    outer_r_alpha = r_alpha.T @ r_alpha
    
    return sum_sx, sum_r_alpha, outer_sx, outer_r_alpha

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling'])
def _compiled_correlation_master(batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
                                 n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
                                 use_noise, use_sampling, cavity_drive):
    j_val = n_spins / 2.0
    
    # EXACTLY 19 arguments for the new Universal Solver signature
    vmap_solver = jax.vmap(
        solve_single_trajectory, 
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry, batch_keys):
        batch_S, batch_alpha = vmap_solver(
            batch_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins, dt, num_steps, 
            Sigma_matrix_weighted, cos_wt, sin_wt, amp, use_noise, use_sampling, cavity_drive, -1, 0.0, 0.0
        )
        
        sum_sx, sum_r_alpha, out_sx, out_r_alpha = _accumulate_correlation_components(batch_S, batch_alpha, j_val)
        return (carry[0] + sum_sx, carry[1] + sum_r_alpha, carry[2] + out_sx, carry[3] + out_r_alpha), None

    init_carry = (
        jnp.zeros(num_steps, dtype=jnp.float64), jnp.zeros(num_steps, dtype=jnp.float64),
        jnp.zeros((num_steps, num_steps), dtype=jnp.float64), jnp.zeros((num_steps, num_steps), dtype=jnp.float64)
    )
    final_carry, _ = jax.lax.scan(master_scan_body, init_carry, batched_keys)
    return final_carry

def calculate_correlations(keys, t_grid, p, B_field=None, w_max=20.0, N_w=5000):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    j_val = p['n_spins'] / 2.0
    
    # --- CRITICAL NEW SANITIZER: GUARANTEES CORRECT B_FIELD SHAPES ---
    if B_field is None:
        B_field = p.get('B_z', 0.0)
        
    B_arr = jnp.asarray(B_field)
    if B_arr.ndim == 0:
        B_field_safe = jnp.zeros((num_steps, 3)).at[:, 2].set(B_arr)
    elif B_arr.shape == (3,):
        B_field_safe = jnp.tile(B_arr, (num_steps, 1))
    else:
        B_field_safe = jnp.reshape(B_arr, (num_steps, 3))
    # -----------------------------------------------------------------

    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(num_steps, dt, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], w_max, N_w)
    Sigma_matrix_weighted, cos_wt, sin_wt, amp = precompute_solver_matrices(num_steps, dt, Sigma_R_t, S_bath_w, dw)

    cavity_drive = jnp.zeros(num_steps, dtype=jnp.float64)
    batched_keys = keys[:(n_total // p['batch_size']) * p['batch_size']].reshape(-1, p['batch_size'], 2)

    print(f"Executing Correlations ({n_total} trajectories) on native GPU...")
    global_sum_sx, global_sum_r_alpha, global_outer_sx, global_outer_r_alpha = _compiled_correlation_master(
        batched_keys, p['omega_0'], B_field_safe, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, True, True, cavity_drive
    )

    mean_sx = global_sum_sx / n_total
    mean_r_alpha = global_sum_r_alpha / n_total
    
    C_spin = (global_outer_sx / n_total) - jnp.outer(mean_sx, mean_sx)
    C_cavity = ((global_outer_r_alpha / n_total) - jnp.outer(mean_r_alpha, mean_r_alpha)) / j_val

    print("Correlations Complete!")
    return np.array(C_spin), np.array(C_cavity)


# =====================================================================
# 2. COMPILED MASTER ENGINE: LINEAR RESPONSES
# =====================================================================

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling'])
def _compiled_response_master(batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
                              n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
                              use_noise, use_sampling, cavity_drive, pulse_idx, epsilon_spin, epsilon_cav):
    
    # EXACTLY 19 arguments
    vmap_solver = jax.vmap(
        solve_single_trajectory, 
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry, batch_keys):
        batch_S, batch_alpha = vmap_solver(
            batch_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, n_spins, dt, num_steps, 
            Sigma_matrix_weighted, cos_wt, sin_wt, amp, use_noise, use_sampling, cavity_drive, pulse_idx, epsilon_spin, epsilon_cav
        )
        return (carry[0] + jnp.sum(batch_S[:, :, 0], axis=0), carry[1] + jnp.sum(jnp.real(batch_alpha), axis=0)), None

    init_carry = (jnp.zeros(num_steps, dtype=jnp.float64), jnp.zeros(num_steps, dtype=jnp.float64))
    final_carry, _ = jax.lax.scan(master_scan_body, init_carry, batched_keys)
    return final_carry

def calculate_responses(keys, t_grid, p, t_pulse, epsilon=0.01, w_max=20.0, N_w=5000):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    pulse_idx = int(np.searchsorted(t_grid, t_pulse))
    j_val = p['n_spins'] / 2.0

    B_base = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[:, 2].set(p.get('B_z', 0.0))
    E_base = jnp.zeros(num_steps, dtype=jnp.float64)

    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(num_steps, dt, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], w_max, N_w)
    Sigma_matrix_weighted, cos_wt, sin_wt, amp = precompute_solver_matrices(num_steps, dt, Sigma_R_t, S_bath_w, dw)
    batched_keys = keys[:(n_total // p['batch_size']) * p['batch_size']].reshape(-1, p['batch_size'], 2)

    print(f"Executing 3-Stage FDT Responses ({n_total} trajectories) on native GPU...")

    sum_base_spin, sum_base_cav = _compiled_response_master(
        batched_keys, p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, True, True, E_base, -1, 0.0, 0.0
    )
    
    sum_pert_spin, _ = _compiled_response_master(
        batched_keys, p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, True, True, E_base, pulse_idx, epsilon, 0.0
    )
    
    _, sum_pert_cav = _compiled_response_master(
        batched_keys, p['omega_0'], B_base, p['g'], p['n_photons_initial'], p['initial_direction'], 
        p['n_spins'], dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, True, True, E_base, pulse_idx, 0.0, epsilon
    )

    response_spin  = ((sum_pert_spin / n_total) - (sum_base_spin / n_total)) / (epsilon * j_val)
    response_cavity = ((sum_pert_cav / n_total) - (sum_base_cav / n_total)) / (epsilon * jnp.sqrt(j_val))

    print("Responses Complete!")
    return np.array(response_spin[pulse_idx:]), np.array(response_cavity[pulse_idx:])

# =====================================================================
# 3. EXACT FOURIER TRANSFORMS
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
def fourier_transform_correlation(c_tau, dt, w_grid, is_spin):
    """
    Original Symmetric Fourier Transform: 
    Direct cosine transform with no padding or windowing.
    """
    tau_grid = jnp.arange(len(c_tau), dtype=jnp.float64) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    weights = jnp.ones_like(c_tau).at[0].set(0.5).at[-1].set(0.5)
    scaling_factor = jnp.where(is_spin, 0.5, 1.0)
    return jnp.dot(jnp.cos(w_tau), c_tau * weights) * dt*scaling_factor

@jax.jit
def fourier_transform_response(chi_tau, dt, w_grid):
    """
    Original Symmetric Fourier Transform: 
    Direct sine transform with no padding or windowing.
    """
    tau_grid = (jnp.arange(len(chi_tau), dtype=jnp.float64) + 0.5) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    weights = jnp.ones_like(chi_tau).at[0].set(0.5).at[-1].set(0.5)
    return jnp.dot(jnp.sin(w_tau), chi_tau * weights) * dt