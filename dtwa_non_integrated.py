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
def compute_explicit_bath_kernels(num_steps, dt, omega_0, alpha, omega_c, s, T, w_max=40.0, N_w=5000):
    w_grid = jnp.linspace(-w_max, w_max, N_w, dtype=jnp.float64)
    dw = w_grid[1] - w_grid[0]
    J_w_pos = jnp.where(w_grid > 1e-10, alpha * omega_c * (w_grid/omega_c)**s * jnp.exp(-w_grid/omega_c), 0.0)
    
    t_grid = jnp.arange(num_steps) * dt
    def compute_sigma(t):
        return -1j * jnp.dot(jnp.exp(-1j * w_grid * t), J_w_pos) * dw
    Sigma_R_t = jax.vmap(compute_sigma)(t_grid)
    
    # [FIX] Removed Sigma_R_t.at[0].multiply(0.5). 
    # The halving is now handled entirely and safely by the local Trapezoidal weights!

    coth_w = 1.0 / jnp.tanh(w_grid / (2.0 * T + 1e-12))
    amp_full = jnp.sqrt(J_w_pos * coth_w * dw / 2.0)
    
    return Sigma_R_t, amp_full, w_grid, dw


@jax.jit(static_argnames=['num_steps'])
def precompute_solver_arrays(num_steps, dt, Sigma_R_t, amp_full, dw, w_grid):
    half_N = amp_full.shape[0] // 2
    w_pos = w_grid[half_N:] 
    amp = amp_full[half_N:]
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    return Sigma_R_t, t_grid, w_pos, amp

# =====================================================================
# 2. HIGH-SPEED TRAJECTORY SOLVERS (PURE HEUN'S METHOD)
# =====================================================================

def non_markovian_coupled_heun_step(S_history, alpha_history, step_idx, xi_curr, xi_next, 
                                   Sigma_R_t, num_steps, B_field_val, coupling_strength, omega_0, dt):
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    alpha_curr = alpha_history[curr_idx]

    # ==========================================================
    # --- EXACT ON-THE-FLY TRAPEZOIDAL ROW GENERATOR ---
    # ==========================================================
    j_idx = jnp.arange(num_steps)
    
    def get_memory(target_idx, alpha_hist):
        weights = jnp.where(j_idx < target_idx, 1.0, 0.0)
        weights = jnp.where(j_idx == 0, 0.5, weights)
        weights = jnp.where(j_idx == target_idx, 0.5, weights)
        weights = jnp.where(target_idx == 0, 0.0, weights)
                
        safe_idx = jnp.where(target_idx >= j_idx, target_idx - j_idx, 0)
        sigma_row = jnp.where(target_idx >= j_idx, Sigma_R_t[safe_idx], 0.0j)
        
        return jnp.dot(sigma_row * weights * dt, alpha_hist)
    # ==========================================================

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
    
    # [FIX] ETD is gone! Full explicit derivative calculation:
    d_alpha_p = -1j * omega_0 * alpha_curr + drive_p
    alpha_pred = alpha_curr + dt * d_alpha_p
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
    
    # [FIX] Corrector combines k1 and k2 directly
    d_alpha_c = -1j * omega_0 * alpha_pred + drive_c
    alpha_next = alpha_curr + 0.5 * dt * (d_alpha_p + d_alpha_c)
    
    return S_history.at[step_idx].set(S_next), alpha_history.at[step_idx].set(alpha_next)


def solve_single_trajectory(key, omega_0, B_field_base, g, n_photons_initial, initial_direction, 
                            n_spins, dt, num_steps, Sigma_R_t, t_grid, w_pos, amp, 
                            use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity):
    k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
    coupling_strength = g / jnp.sqrt(n_spins)
    
    s0_sampled = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins) / 2.0
    s0_mean = (initial_direction * n_spins) / 2.0
    s0 = jnp.where(use_sampling, s0_sampled, s0_mean)
    
    alpha0_sampled = cavity_wigner_sampling(k_samp_alpha, n_photons_initial)
    alpha0_mean = jnp.sqrt(jnp.array(n_photons_initial, dtype=jnp.float64)) + 0j
    alpha0 = jnp.where(use_sampling, alpha0_sampled, alpha0_mean)
    
    S_history = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[0].set(s0)
    alpha_history = jnp.zeros((num_steps,), dtype=jnp.complex128).at[0].set(alpha0)
    
    half_N = amp.shape[0]
    k_re, k_im = jax.random.split(k_noise)
    noise_re = jax.random.normal(k_re, (half_N,), dtype=jnp.float64) * amp
    noise_im = jax.random.normal(k_im, (half_N,), dtype=jnp.float64) * amp

    def compute_noise(t):
        wt = t * w_pos
        xi_real = jnp.dot(jnp.cos(wt), noise_re) + jnp.dot(jnp.sin(wt), noise_im)
        xi_imag = jnp.dot(jnp.cos(wt), noise_im) - jnp.dot(jnp.sin(wt), noise_re)
        return jnp.where(use_noise, xi_real + 1j * xi_imag, 0j)

    xi_0 = compute_noise(t_grid[0])
    
    def scan_body(carry, step_idx):
        S_hist, alpha_hist, xi_curr = carry
        curr_idx = step_idx - 1
        
        B_val_step = jax.lax.dynamic_index_in_dim(B_field_base, curr_idx, axis=0, keepdims=False)
        xi_next = compute_noise(t_grid[step_idx])
        
        # [FIX] Swapped ETD for pure Heun predictor-corrector
        S_hist_updated, alpha_hist_updated = non_markovian_coupled_heun_step(
            S_hist, alpha_hist, step_idx, xi_curr, xi_next, Sigma_R_t, num_steps, 
            B_val_step, coupling_strength, omega_0, dt
        )
        
        current_S = S_hist_updated[step_idx]
        current_alpha = alpha_hist_updated[step_idx]
        
        # ==========================================================
        # EXACT ANALYTICAL JUMPS
        # ==========================================================
        is_pulse = (step_idx == pulse_idx)
        cos_e = jnp.cos(epsilon_spin)
        sin_e = jnp.sin(epsilon_spin)
        
        S_y_jumped = current_S[1] * cos_e - current_S[2] * sin_e
        S_z_jumped = current_S[2] * cos_e + current_S[1] * sin_e
        S_jumped = jnp.array([current_S[0], S_y_jumped, S_z_jumped])
        
        current_S = jnp.where(is_pulse, S_jumped, current_S)
        alpha_jumped = current_alpha + 1j * epsilon_cavity
        current_alpha = jnp.where(is_pulse, alpha_jumped, current_alpha)
        # ==========================================================
        
        S_hist_final = S_hist_updated.at[step_idx].set(current_S)
        alpha_hist_final = alpha_hist_updated.at[step_idx].set(current_alpha)
        
        return (S_hist_final, alpha_hist_final, xi_next), None

    init_carry = (S_history, alpha_history, xi_0)
    time_indices = jnp.arange(1, num_steps, dtype=jnp.int64)
    (final_S_history, final_alpha_history, _), _ = jax.lax.scan(scan_body, init_carry, time_indices)
    
    return final_S_history, final_alpha_history


# =====================================================================
# 3. GLOBAL BATCH COMPILER & WRAPPER
# =====================================================================

@jax.jit
def _accumulate_batch_sums(spin_ensemble, cavity_ensemble, j_val):
    jx_trajs = spin_ensemble[:, :, 0] / j_val
    jy_trajs = spin_ensemble[:, :, 1] / j_val
    jz_trajs = spin_ensemble[:, :, 2] / j_val

    return {
        "sum_jx": jnp.sum(jx_trajs, axis=0),
        "sum_jy": jnp.sum(jy_trajs, axis=0),
        "sum_jz": jnp.sum(jz_trajs, axis=0),
        "sum_jx_sq": jnp.sum(jx_trajs**2, axis=0),
        "sum_abs_jx": jnp.sum(jnp.abs(jx_trajs), axis=0),
        "sum_psi": jnp.sum(cavity_ensemble, axis=0),
        "sum_psi_sq": jnp.sum(jnp.abs(cavity_ensemble)**2, axis=0),
        "outer_sx": (spin_ensemble[:, :, 0] / j_val).T @ (spin_ensemble[:, :, 0] / j_val),
        "outer_r_alpha": jnp.real(cavity_ensemble).T @ jnp.real(cavity_ensemble)
    }

@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling'])
def _compiled_master_processor(batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
                               n_spins, dt, num_steps, Sigma_R_t, t_grid, w_pos, amp, 
                               use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity):
    
    j_val = n_spins / 2.0
    vmap_solver = jax.vmap(
        solve_single_trajectory, 
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry_stats, current_batch_keys):
        batch_S, batch_alpha = vmap_solver(
            current_batch_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
            n_spins, dt, num_steps, Sigma_R_t, t_grid, w_pos, amp, 
            use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity
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
        "outer_sx": jnp.zeros((num_steps, num_steps), dtype=jnp.float64),
        "outer_r_alpha": jnp.zeros((num_steps, num_steps), dtype=jnp.float64)
    }
    
    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats


def run_dtwa(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, 
             batch_size=1000, n_spins=1, w_max=40.0, N_w=5000, use_noise=True, use_sampling=True, 
             pulse_idx=-1, epsilon_spin=0.0, epsilon_cavity=0.0):
    
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

    Sigma_R_t, t_grid_pre, w_pos, amp = precompute_solver_arrays(
        num_steps, dt, Sigma_R_t, amp_full, dw, w_grid)

    n_batches = n_total // batch_size
    batched_keys = keys[:n_batches * batch_size].reshape(n_batches, batch_size, -1)

    print(f"Executing {n_total} trajectories across {n_batches} compiled batches...")
    
    running_stats = _compiled_master_processor(
        batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
        n_spins, dt, num_steps, Sigma_R_t, t_grid_pre, w_pos, amp, 
        use_noise, use_sampling, pulse_idx, epsilon_spin, epsilon_cavity
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