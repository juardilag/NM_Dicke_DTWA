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
# 1. KERNELS & PRECOMPUTE ENGINE (Maximum Precision & Speed)
# =====================================================================

@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_explicit_bath_kernels(num_steps, dt, omega_0, alpha, omega_c, s, T, w_max=20.0, N_w=5000):
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    w_grid = jnp.linspace(-w_max, w_max, N_w, dtype=jnp.float64)
    dw = w_grid[1] - w_grid[0]
    
    abs_w = jnp.abs(w_grid)
    J_w = jnp.where(w_grid != 0.0, jnp.sign(w_grid) * alpha * omega_c * (abs_w/omega_c)**s * jnp.exp(-abs_w/omega_c), 0.0)
    
    w_diff = w_grid[:, None] - w_grid[None, :]
    mask = jnp.eye(N_w, dtype=jnp.float64)
    pv_kernel = (1.0 - mask) / (w_diff + mask) 
    
    # PRECISION UPGRADE: Trapezoidal integration weights for spectral transform
    trapz_w = jnp.ones(N_w, dtype=jnp.float64).at[0].set(0.5).at[-1].set(0.5)
    
    Sigma_Real = jnp.dot(pv_kernel, J_w * trapz_w) * dw / jnp.pi
    Sigma_R_w = Sigma_Real - 1j * jnp.pi * J_w 
    
    wt = t_grid[:, None] * w_grid[None, :]
    
    # exp(-1j*wt) is faster to compile and mathematically identical to cos - 1j*sin
    Sigma_R_t = jnp.dot(jnp.exp(-1j * wt), Sigma_R_w * trapz_w) * dw / (2.0 * jnp.pi)
    
    S_bath_w = jnp.pi * jnp.abs(J_w) * jnp.where(abs_w > 1e-10, 1.0 / jnp.tanh(abs_w / (2.0 * T + 1e-12)), 0.0)
    
    return Sigma_R_t, S_bath_w, w_grid, dw


@jax.jit(static_argnames=['num_steps'])
def precompute_solver_matrices(num_steps, dt, Sigma_R_t, S_bath_w, dw):
    """
    Bakes all O(N^2) masks, trigonometric evaluations, and numerical integration
    weights into static matrices outside the trajectory loops.
    """
    # 1. Non-Markovian Memory Matrix with Trapezoidal Weights
    i_idx = jnp.arange(num_steps)[:, None]
    j_idx = jnp.arange(num_steps)[None, :]
    
    # Sigma_matrix[i, j] = Sigma_R_t[i - j] if i >= j else 0
    Sigma_matrix = jnp.where(i_idx >= j_idx, Sigma_R_t[i_idx - j_idx], 0.0 + 0j)
    
    # Trapezoidal rule weights: 0.5 at endpoints, 1.0 in the middle
    trapz_weights = jnp.where((j_idx == 0) | (j_idx == i_idx), 0.5,
                    jnp.where((j_idx > 0) & (j_idx < i_idx), 1.0, 0.0))
    
    Sigma_matrix_weighted = Sigma_matrix * trapz_weights * dt

    # 2. Stochastic Noise Trigonometric Matrices
    half_N = S_bath_w.shape[0] // 2
    w_pos = jnp.linspace(0.0, 20.0, half_N, dtype=jnp.float64)
    S_pos = S_bath_w[half_N:]
    
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    wt_pos = t_grid[:, None] * w_pos[None, :]
    
    # Pre-evaluate transcendental functions globally
    cos_wt = jnp.cos(wt_pos)
    sin_wt = jnp.sin(wt_pos)
    amp = jnp.sqrt(S_pos * dw / jnp.pi)

    return Sigma_matrix_weighted, cos_wt, sin_wt, amp

# =====================================================================
# 2. HIGH-SPEED TRAJECTORY SOLVERS
# =====================================================================

def generate_explicit_bath_noise(key, cos_wt, sin_wt, amp, use_noise=True):
    """Generates noise via hyper-fast O(1) matrix multiplication."""
    half_N = amp.shape[0]
    k_re, k_im = jax.random.split(key)
    
    # Apply amplitudes immediately to 1D arrays
    noise_re = jax.random.normal(k_re, (half_N,), dtype=jnp.float64) * amp
    noise_im = jax.random.normal(k_im, (half_N,), dtype=jnp.float64) * amp
    
    xi_t_real = jnp.where(use_noise, 
                          jnp.dot(cos_wt, noise_re) - jnp.dot(sin_wt, noise_im), 
                          0.0)
    return xi_t_real


def non_markovian_coupled_etd_step(S_history, alpha_history, step_idx, noise_traj, Sigma_matrix_weighted, 
                                   B_field_val, coupling_strength, omega_0, dt):
    """
    Step calculation optimized. Replaced dynamic masks with O(1) pre-sliced matrix dot products.
    """
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    alpha_curr = alpha_history[curr_idx]
    
    z = 1j * omega_0
    exact_decay = jnp.exp(-z * dt)
    phi_drive = (1.0 - exact_decay) / z

    # --- 1. PREDICTOR ---
    # PRECISION/SPEED UPGRADE: Exact trapezoidal integration via precomputed matrix slice
    memory_p = jnp.dot(Sigma_matrix_weighted[curr_idx], alpha_history)
    
    B_eff_p_x = 0.5 * B_field_val[0] + 2.0 * coupling_strength * jnp.real(alpha_curr)
    B_eff_p = jnp.array([B_eff_p_x, 0.5 * B_field_val[1], 0.5 * B_field_val[2]], dtype=jnp.float64)
    
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
    
    drive_p = -1j * memory_p - 1j * coupling_strength * (2.0 * S_curr[0]) + 1j * noise_traj[curr_idx]
    alpha_pred = alpha_curr * exact_decay + drive_p * phi_drive
    
    alpha_history_pred = alpha_history.at[step_idx].set(alpha_pred)
    
    # --- 2. CORRECTOR ---
    memory_c = jnp.dot(Sigma_matrix_weighted[step_idx], alpha_history_pred)
    
    B_eff_c_x = 0.5 * B_field_val[0] + 2.0 * coupling_strength * jnp.real(alpha_pred)
    B_eff_c = jnp.array([B_eff_c_x, 0.5 * B_field_val[1], 0.5 * B_field_val[2]], dtype=jnp.float64)
    
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
    
    drive_c = -1j * memory_c - 1j * coupling_strength * (2.0 * S_pred[0]) + 1j * noise_traj[step_idx]
    alpha_next = alpha_curr * exact_decay + 0.5 * (drive_p + drive_c) * phi_drive
    
    return S_history.at[step_idx].set(S_next), alpha_history.at[step_idx].set(alpha_next)


def solve_single_trajectory(key, omega_0, B_field, g, n_photons_initial, initial_direction, 
                            n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
                            use_noise, use_sampling, cavity_drive, pulse_idx=-1, epsilon_spin=0.0, epsilon_cav=0.0):
    
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
    
    noise_traj_real = generate_explicit_bath_noise(k_noise, cos_wt, sin_wt, amp, use_noise=use_noise)
    noise_traj_complex = noise_traj_real + 0j
    
    def scan_body(carry, step_idx):
        S_hist, alpha_hist = carry
        curr_idx = step_idx - 1
        
        # 1. Dynamically slice arrays at the exact time step
        B_val_step = jax.lax.dynamic_index_in_dim(B_field, curr_idx, axis=0, keepdims=False)
        E_val_step = jax.lax.dynamic_index_in_dim(cavity_drive, curr_idx, axis=0, keepdims=False)
        
        # 2. Standard Unperturbed Evolution
        # CAUGHT THE BUG: This returns the FULL (5000, 3) updated history arrays!
        S_hist_updated, alpha_hist_updated = non_markovian_coupled_etd_step(
            S_hist, alpha_hist, step_idx, noise_traj_complex, Sigma_matrix_weighted, 
            B_val_step, coupling_strength, omega_0, dt
        )
        
        # Extract strictly the 1D vectors for the current step
        current_S = S_hist_updated[step_idx]
        current_alpha = alpha_hist_updated[step_idx]
        
        # Apply standard continuous cavity drive (if any)
        z = 1j * omega_0
        phi_drive = (1.0 - jnp.exp(-z * dt)) / z
        current_alpha = current_alpha - 1j * E_val_step * phi_drive
        
        # 3. The EXACT Instantaneous Dirac Kicks (Only triggers at pulse_idx)
        is_pulse = (step_idx == pulse_idx)
        
        cos_e, sin_e = jnp.cos(epsilon_spin), jnp.sin(epsilon_spin)
        S_kicked = jnp.array([
            current_S[0],
            current_S[1] * cos_e - current_S[2] * sin_e,
            current_S[2] * cos_e + current_S[1] * sin_e
        ], dtype=jnp.float64)
        
        # Snap the 1D vectors if pulse is triggered
        final_S_step = jnp.where(is_pulse, S_kicked, current_S)
        final_alpha_step = jnp.where(is_pulse, current_alpha - 1j * epsilon_cav, current_alpha)
        
        # Safely overwrite the history matrix with the finalized step
        S_hist_final = S_hist_updated.at[step_idx].set(final_S_step)
        alpha_hist_final = alpha_hist_updated.at[step_idx].set(final_alpha_step)
        
        return (S_hist_final, alpha_hist_final), None

    init_carry = (S_history, alpha_history)
    time_indices = jnp.arange(1, num_steps, dtype=jnp.int64)
    (final_S_history, final_alpha_history), _ = jax.lax.scan(scan_body, init_carry, time_indices)
    
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
        "sum_psi_sq": jnp.sum(jnp.abs(cavity_ensemble)**2, axis=0)
    }


@jax.jit(static_argnames=['n_spins', 'num_steps', 'use_noise', 'use_sampling'])
def _compiled_master_processor(batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
                               n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
                               use_noise, use_sampling, cavity_drive):
    
    j_val = n_spins / 2.0
    
    # Safe vmap wrapping ensuring exactly 19 arguments
    vmap_solver = jax.vmap(
        solve_single_trajectory, 
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    )

    def master_scan_body(carry_stats, current_batch_keys):
        # Pass normal physics triggers (-1 for pulse_idx and 0.0 for kicks)
        batch_S, batch_alpha = vmap_solver(
            current_batch_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
            n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
            use_noise, use_sampling, cavity_drive, -1, 0.0, 0.0
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
        "sum_psi_sq": jnp.zeros(num_steps, dtype=jnp.float64)
    }
    
    final_running_stats, _ = jax.lax.scan(master_scan_body, init_stats, batched_keys)
    return final_running_stats


def run_dtwa(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, 
             batch_size=1000, n_spins=1, w_max=20.0, N_w=5000, use_noise=True, use_sampling=True, cavity_drive=None):
    
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # --- CRITICAL NEW SANITIZER: GUARANTEES CORRECT B_FIELD SHAPES ---
    B_arr = jnp.asarray(B_field)
    if B_arr.ndim == 0:
        # If user passed a single scalar number (B_z)
        B_field_safe = jnp.zeros((num_steps, 3)).at[:, 2].set(B_arr)
    elif B_arr.shape == (3,):
        # If user passed a constant vector [0.0, 0.0, B_z]
        B_field_safe = jnp.tile(B_arr, (num_steps, 1))
    else:
        # Strip any accidental batch dimensions and force to (num_steps, 3)
        B_field_safe = jnp.reshape(B_arr, (num_steps, 3))
    # -----------------------------------------------------------------
    
    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(
        num_steps, dt, omega_0, alpha, omega_c, s, T, w_max, N_w)

    Sigma_matrix_weighted, cos_wt, sin_wt, amp = precompute_solver_matrices(
        num_steps, dt, Sigma_R_t, S_bath_w, dw)

    if cavity_drive is None:
        cavity_drive = jnp.zeros(num_steps, dtype=jnp.float64)

    n_batches = n_total // batch_size
    batched_keys = keys[:n_batches * batch_size].reshape(n_batches, batch_size, -1)

    print(f"Executing {n_total} trajectories across {n_batches} compiled batches...")
    
    running_stats = _compiled_master_processor(
        batched_keys, omega_0, B_field_safe, g, n_photons_initial, initial_direction, 
        n_spins, dt, num_steps, Sigma_matrix_weighted, cos_wt, sin_wt, amp, 
        use_noise, use_sampling, cavity_drive
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