import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"

import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import jax.debug
from initial_samplings import discrete_spin_sampling_factorized, cavity_wigner_sampling


@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_explicit_bath_kernels(num_steps, dt, omega_0, alpha, omega_c, s, T, w_max=20.0, N_w=5000):
    """
    Computes the time-domain bath memory kernel and bare noise power spectrum.
    Optimized to minimize memory operations.
    """
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    w_grid = jnp.linspace(-w_max, w_max, N_w, dtype=jnp.float64)
    dw = w_grid[1] - w_grid[0]
    
    abs_w = jnp.abs(w_grid)
    # Replaced jnp.where condition to safely prevent division by zero while preserving float64
    J_w = jnp.where(w_grid != 0.0, jnp.sign(w_grid) * alpha * omega_c * (abs_w/omega_c)**s * jnp.exp(-abs_w/omega_c), 0.0)
    
    w_diff = w_grid[:, None] - w_grid[None, :]
    mask = jnp.eye(N_w, dtype=jnp.float64)
    pv_kernel = (1.0 - mask) / (w_diff + mask) 
    Sigma_Real = jnp.dot(pv_kernel, J_w) * dw / jnp.pi
    
    Sigma_R_w = Sigma_Real - 1j * jnp.pi * J_w 
    
    wt = t_grid[:, None] * w_grid[None, :]
    Sigma_R_t = jnp.dot(jnp.cos(wt) - 1j * jnp.sin(wt), Sigma_R_w) * dw / (2.0 * jnp.pi)
    
    S_bath_w = jnp.pi * jnp.abs(J_w) * jnp.where(abs_w > 1e-10, 1.0 / jnp.tanh(abs_w / (2.0 * T + 1e-12)), 0.0)
    
    return Sigma_R_t, S_bath_w, w_grid, dw


@jax.jit(static_argnames=['num_steps'])
def generate_explicit_bath_noise(key, num_steps, dt, S_bath_w, w_grid, dw, use_noise=True):
    half_N = S_bath_w.shape[0] // 2
    w_pos = jnp.linspace(0.0, 20.0, half_N, dtype=jnp.float64)
    S_pos = S_bath_w[half_N:]
    
    k_re, k_im = jax.random.split(key)
    amp = jnp.sqrt(S_pos * dw / jnp.pi)
    
    t_grid = jnp.arange(num_steps, dtype=jnp.float64) * dt
    wt_pos = t_grid[:, None] * w_pos[None, :]
    
    noise_re = jax.random.normal(k_re, (half_N,), dtype=jnp.float64)
    noise_im = jax.random.normal(k_im, (half_N,), dtype=jnp.float64)
    
    xi_t = jnp.where(use_noise, 
                     jnp.dot(jnp.cos(wt_pos), noise_re * amp) + \
                     1j * jnp.dot(jnp.sin(wt_pos), noise_im * amp), 
                     0.0 + 0j)
    return xi_t


@jax.jit(static_argnames=['num_steps'])
def non_markovian_coupled_etd_step(S_history, alpha_history, step_idx, noise_traj, Sigma_R_t, B_field_val, 
                                   coupling_strength, omega_0, dt, num_steps):
    """
    Optimized step calculation. Removed costly structural O(N^2) masks inside the loop.
    """
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    alpha_curr = alpha_history[curr_idx]
    
    z = 1j * omega_0
    exact_decay = jnp.exp(-z * dt)
    phi_drive = (1.0 - exact_decay) / z

    # --- 1. PREDICTOR ---
    # Optimized: Instead of creating a dynamic full-sized mask, use static roll-back lookup indices
    steps = jnp.arange(num_steps, dtype=jnp.int64)
    kernel_p = jnp.where(steps < step_idx, Sigma_R_t[jnp.maximum(0, curr_idx - steps)], 0.0 + 0j)
    memory_p = jnp.dot(kernel_p, alpha_history) * dt
    
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
    kernel_c = jnp.where(steps <= step_idx, Sigma_R_t[jnp.maximum(0, step_idx - steps)], 0.0 + 0j)
    memory_c = jnp.dot(kernel_c, alpha_history_pred) * dt
    
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


def solve_single_trajectory(key, t_grid, omega_0, B_field, g, n_photons_initial, initial_direction, 
                             n_spins, dt, num_steps, Sigma_R_t, S_bath_w, w_grid, dw, 
                             use_noise, use_sampling, cavity_drive):
    """
    Unified trajectory solver. Fixed tracing bugs by using native JAX dynamic 
    index lookup functions instead of Python indices.
    """
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
    
    noise_traj = generate_explicit_bath_noise(k_noise, num_steps, dt, S_bath_w, w_grid, dw, use_noise=use_noise)
    
    def scan_body(carry, step_idx):
        S_hist, alpha_hist = carry
        
        # FIX: Safe dynamic tracing indexing via dynamic_index_in_dim
        B_val = jax.lax.dynamic_index_in_dim(B_field, step_idx, axis=0, keepdims=False)
        E_val = jax.lax.dynamic_index_in_dim(cavity_drive, step_idx, axis=0, keepdims=False)
        
        S_next_hist, alpha_next_hist = non_markovian_coupled_etd_step(
            S_hist, alpha_hist, step_idx, noise_traj, Sigma_R_t, B_val, 
            coupling_strength, omega_0, dt, num_steps
        )
        
        z = 1j * omega_0
        phi_drive = (1.0 - jnp.exp(-z * dt)) / z
        alpha_next_hist = alpha_next_hist.at[step_idx].add(-1j * E_val * phi_drive)
        
        return (S_next_hist, alpha_next_hist), None

    init_carry = (S_history, alpha_history)
    time_indices = jnp.arange(1, num_steps, dtype=jnp.int64)
    (final_S_history, final_alpha_history), _ = jax.lax.scan(scan_body, init_carry, time_indices)
    
    return final_S_history, final_alpha_history

@jax.jit
def _accumulate_batch_sums(spin_ensemble, cavity_ensemble, j_val):
    """
    Computes sums of observables for a single batch natively on the device.
    """
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


def run_dtwa(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, 
                                          batch_size=1000, n_spins=1, w_max=20.0, N_w=5000, use_noise=True, use_sampling=True, cavity_drive=None):
    """
    Runs trajectories in batches, immediately reducing them to statistical summaries.
    Guarantees flat O(1) memory usage relative to total trajectory count.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(
        num_steps, dt, omega_0, alpha, omega_c, s, T, w_max, N_w)

    if cavity_drive is None:
        cavity_drive = jnp.zeros(num_steps, dtype=jnp.float64)

    n_batches = int(jnp.ceil(n_total / batch_size))
    pbar = tqdm(total=n_batches, desc=f"Running DTWA of {n_total} trajectories in {n_batches} batches")

    # 2. Modern callback functions do not take hidden transformation arguments
    def update_pbar():
        pbar.update(1)

    @jax.jit
    def process_and_reduce_batch(batch_keys):
        # 1. Run trajectories for this batch
        batch_S, batch_alpha = jax.vmap(lambda k: solve_single_trajectory(
            k, t_grid, omega_0, B_field, g, n_photons_initial, initial_direction, 
            n_spins, dt, num_steps, Sigma_R_t, S_bath_w, w_grid, dw, 
            use_noise, use_sampling, cavity_drive
        ))(batch_keys)
        
        # 2. Immediately reduce batch to 1D statistical components
        batch_sums = _accumulate_batch_sums(batch_S, batch_alpha, n_spins)
        
        # 3. Use the modern debug callback to trigger the progress bar tick
        jax.debug.callback(update_pbar)
        return batch_sums

    # Initialize empty tracking arrays on Host CPU RAM to hold cumulative statistics
    running_stats = {
        "sum_jx": np.zeros(num_steps, dtype=np.float64),
        "sum_jy": np.zeros(num_steps, dtype=np.float64),
        "sum_jz": np.zeros(num_steps, dtype=np.float64),
        "sum_jx_sq": np.zeros(num_steps, dtype=np.float64),
        "sum_abs_jx": np.zeros(num_steps, dtype=np.float64),
        "sum_psi": np.zeros(num_steps, dtype=np.complex128),
        "sum_psi_sq": np.zeros(num_steps, dtype=np.float64)
    }
    
    try:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_total)
            current_keys = keys[start_idx:end_idx]
            
            # Compute current batch reductions on device
            batch_sums = process_and_reduce_batch(current_keys)
            
            # Pull down the tiny 1D arrays to CPU and accumulate immediately
            for key in running_stats:
                batch_sums[key].block_until_ready()
                running_stats[key] += np.array(batch_sums[key])
                
    finally:
        pbar.close()

    # Final normalization step over the entire global ensemble size (n_total)
    final_stats = {
        "j_x": running_stats["sum_jx"] / n_total,
        "j_y": running_stats["sum_jy"] / n_total,
        "j_z": running_stats["sum_jz"] / n_total,
        "rms_jx": np.sqrt(running_stats["sum_jx_sq"] / n_total),
        "abs_jx": running_stats["sum_abs_jx"] / n_total,
        "mean_psi": running_stats["sum_psi"] / n_total,
        "abs_mean_psi": np.abs(running_stats["sum_psi"] / n_total),
        "mean_photon_number": (running_stats["sum_psi_sq"] / n_total) - 0.5
    }
    
    # Clean up device compilation memory
    jax.clear_caches()
    
    return final_stats