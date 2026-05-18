import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized, cavity_wigner_sampling
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_explicit_bath_kernels(num_steps, dt, omega_0, alpha, omega_c, s, T, w_max=20.0, N_w=5000):
    """
    Computes the time-domain bath memory kernel and bare noise power spectrum
    using the exact same full-grid parameters as your integrated code.
    """
    t_grid = jnp.arange(num_steps) * dt
    w_grid = jnp.linspace(-w_max, w_max, N_w)
    dw = w_grid[1] - w_grid[0]
    
    abs_w = jnp.abs(w_grid)
    J_w = jnp.where(w_grid != 0, jnp.sign(w_grid) * alpha * omega_c * (abs_w/omega_c)**s * jnp.exp(-abs_w/omega_c), 0.0)
    
    # Manual Kramers-Kronig Principal Value Integral
    w_diff = w_grid[:, None] - w_grid[None, :]
    mask = jnp.eye(N_w)
    pv_kernel = (1.0 - mask) / (w_diff + mask) 
    Sigma_Real = jnp.dot(pv_kernel, J_w) * dw / jnp.pi
    
    # Full Retarded Self-Energy of the bare bath
    Sigma_R_w = Sigma_Real - 1j * jnp.pi * J_w 
    
    # Inverse Fourier Transform to obtain the complex time-domain cavity memory kernel
    wt = t_grid[:, None] * w_grid[None, :]
    Sigma_R_t = jnp.dot(jnp.cos(wt) - 1j * jnp.sin(wt), Sigma_R_w) * dw / (2.0 * jnp.pi)
    
    # Bare Bath Noise Power Spectrum via Fluctuation-Dissipation Theorem
    S_bath_w = jnp.pi * jnp.abs(J_w) * jnp.where(abs_w > 1e-10, 1.0 / jnp.tanh(abs_w / (2.0 * T + 1e-12)), 0.0)
    
    return Sigma_R_t, S_bath_w, w_grid, dw


@jax.jit(static_argnames=['num_steps'])
def generate_explicit_bath_noise(key, num_steps, dt, S_bath_w, w_grid, dw, use_noise=True):
    half_N = S_bath_w.shape[0] // 2
    w_pos = jnp.linspace(0, 20.0, half_N)
    S_pos = S_bath_w[half_N:]
    
    k_re, k_im = jax.random.split(key)
    amp = jnp.sqrt(S_pos * dw / jnp.pi)
    
    t_grid = jnp.arange(num_steps) * dt
    wt_pos = t_grid[:, None] * w_pos[None, :]
    
    noise_re = jax.random.normal(k_re, (half_N,))
    noise_im = jax.random.normal(k_im, (half_N,))
    
    # Complex noise acting directly on the non-Hermitian cavity quadratures
    xi_t = jnp.where(use_noise, 
                     jnp.dot(jnp.cos(wt_pos), noise_re * amp) + \
                     1j * jnp.dot(jnp.sin(wt_pos), noise_im * amp), 
                     0.0 + 0j)
    return xi_t


@jax.jit(static_argnames=['num_steps'])
def non_markovian_coupled_etd_step(S_history, alpha_history, step_idx, noise_traj, Sigma_R_t, B_field_val, g, omega_0, n_spins, dt, num_steps):
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    alpha_curr = alpha_history[curr_idx]
    
    coupling_strength = g / jnp.sqrt(n_spins)
    
    # Exact Linear Propagators for the bare cavity frequency
    z = 1j * omega_0
    exact_decay = jnp.exp(-z * dt)
    phi_drive = (1.0 - exact_decay) / z

    # --- 1. PREDICTOR ---
    # Memory convolution over the history of the cavity field
    mask_p = jnp.arange(num_steps) < step_idx
    lag_p = curr_idx - jnp.arange(num_steps)
    kernel_p = jnp.where(mask_p, Sigma_R_t[jnp.maximum(0, lag_p)], 0.0 + 0j)
    memory_p = jnp.dot(kernel_p, alpha_history) * dt
    
    # Semiclassical effective magnetic field matching your integrated torque setup
    B_eff_p_x = 0.5 * B_field_val[0] + 2.0 * coupling_strength * jnp.real(alpha_curr)
    B_eff_p = jnp.array([B_eff_p_x, 0.5 * B_field_val[1], 0.5 * B_field_val[2]])
    
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
    
    # Cavity explicit drive (S is divided by 2, so J_x = 2 * S_x)
    drive_p = -1j * memory_p - 1j * coupling_strength * (2.0 * S_curr[0]) + 1j * noise_traj[curr_idx]
    alpha_pred = alpha_curr * exact_decay + drive_p * phi_drive
    
    alpha_history_pred = alpha_history.at[step_idx].set(alpha_pred)
    
    # --- 2. CORRECTOR ---
    mask_c = jnp.arange(num_steps) <= step_idx
    lag_c = step_idx - jnp.arange(num_steps)
    kernel_c = jnp.where(mask_c, Sigma_R_t[jnp.maximum(0, lag_c)], 0.0 + 0j)
    memory_c = jnp.dot(kernel_c, alpha_history_pred) * dt
    
    B_eff_c_x = 0.5 * B_field_val[0] + 2.0 * coupling_strength * jnp.real(alpha_pred)
    B_eff_c = jnp.array([B_eff_c_x, 0.5 * B_field_val[1], 0.5 * B_field_val[2]])
    
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


def run_coupled_non_markovian_twa_bundle(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, 
                                         batch_size=1000, n_spins=1, w_max=20.0, N_w=5000, use_noise=True, use_sampling=True, cavity_drive=None):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(
        num_steps, dt, omega_0, alpha, omega_c, s, T, w_max, N_w)

    # Initialize an empty drive array if none is provided to keep JAX operations smooth
    if cavity_drive is None:
        cavity_drive = jnp.zeros(num_steps)

    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        s0_sampled = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins) / 2.0
        s0_mean = (initial_direction * n_spins) / 2.0
        s0 = jnp.where(use_sampling, s0_sampled, s0_mean)
        
        alpha0_sampled = cavity_wigner_sampling(k_samp_alpha, n_photons_initial)
        alpha0_mean = jnp.sqrt(jnp.array(n_photons_initial, dtype=jnp.float64)) + 0j
        alpha0 = jnp.where(use_sampling, alpha0_sampled, alpha0_mean)
        
        S_history = jnp.zeros((num_steps, 3)).at[0].set(s0)
        alpha_history = jnp.zeros((num_steps,), dtype=jnp.complex128).at[0].set(alpha0)
        
        noise_traj = generate_explicit_bath_noise(
            k_noise, num_steps, dt, S_bath_w, w_grid, dw, use_noise=use_noise)
        
        def scan_body(carry, step_idx):
            S_hist, alpha_hist = carry
            B_val = B_field[step_idx] 
            
            # --- EXTRACT THE EXTERNAL PROBE DRIVE VAL ---
            E_val = cavity_drive[step_idx - 1]
            
            # We add -1j * E_val to the drive parameters passed downstream
            S_next_hist, alpha_next_hist = non_markovian_coupled_etd_step(
                S_hist, alpha_hist, step_idx, noise_traj, Sigma_R_t, B_val, g, omega_0, n_spins, dt, num_steps)
            
            # Directly add the exact linear ETD impulse contribution of the probe to alpha
            z = 1j * omega_0
            phi_drive = (1.0 - jnp.exp(-z * dt)) / z
            alpha_next_hist = alpha_next_hist.at[step_idx].add(-1j * E_val * phi_drive)
            
            return (S_next_hist, alpha_next_hist), None

        init_carry = (S_history, alpha_history)
        (final_S_history, final_alpha_history), _ = jax.lax.scan(scan_body, init_carry, jnp.arange(1, num_steps))
        
        return final_S_history, final_alpha_history

    @jax.jit
    def process_batch(batch_keys):
        return jax.vmap(solve_single_trajectory)(batch_keys)

    all_S, all_alpha = [], []
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_S, batch_alpha = process_batch(current_keys)
        all_S.append(batch_S)
        all_alpha.append(batch_alpha)
        
    return jnp.concatenate(all_S, axis=0), jnp.concatenate(all_alpha, axis=0)


# Extraction of observables, correlations and responses. 

def get_stats(spin_ensemble, cavity_ensemble, j_val):
    """
    Extracts intensive trajectories and order parameters.
    Takes full advantage of the 3D and 2D uncollapsed ensemble formats.
    """
    jx_trajs = spin_ensemble[:, :, 0] / j_val
    jy_trajs = spin_ensemble[:, :, 1] / j_val
    jz_trajs = spin_ensemble[:, :, 2] / j_val
    
    mean_jx = jnp.mean(jx_trajs, axis=0)
    mean_jy = jnp.mean(jy_trajs, axis=0)
    mean_jz = jnp.mean(jz_trajs, axis=0)
    
    rms_jx = jnp.sqrt(jnp.mean(jx_trajs**2, axis=0))
    abs_jx = jnp.mean(jnp.abs(jx_trajs), axis=0)
    
    mean_psi = jnp.mean(cavity_ensemble, axis=0)
    abs_mean_psi = jnp.abs(mean_psi)
    mean_photon_number = jnp.mean(jnp.abs(cavity_ensemble)**2, axis=0) - 0.5
    
    return {
        "j_x": mean_jx,
        "j_y": mean_jy,
        "j_z": mean_jz,
        "rms_jx": rms_jx,
        "abs_jx": abs_jx,
        "mean_psi": mean_psi,
        "abs_mean_psi": abs_mean_psi,
        "mean_photon_number": mean_photon_number
    }