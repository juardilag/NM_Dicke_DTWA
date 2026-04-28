import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_non_markovian_bath_functions(num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins, w_max=20.0, N_w=2000):
    """
    Computes the exact non-Markovian memory kernel and noise power spectrum using Riemann Sums.
    Pre-computes the continuous time-frequency transformation matrices for XLA optimization.
    """
    t_grid = jnp.arange(num_steps) * dt
    
    # 1. Frequency grid for the Riemann sum
    w_grid = jnp.linspace(-w_max, w_max, N_w)
    dw = w_grid[1] - w_grid[0]
    abs_w = jnp.abs(w_grid)
    
    # 2. Spectral Density J(w)
    J_w = jnp.where(w_grid != 0, jnp.sign(w_grid) * alpha * omega_c * (abs_w / omega_c)**s * jnp.exp(-abs_w / omega_c), 0.0)
    
    # 3. Retarded Cavity Green's Function
    Sigma_R = -1j * jnp.pi * J_w 
    chi_R_w = 1.0 / (w_grid - omega_0 - Sigma_R + 1e-10j) - 1.0 / (w_grid + omega_0 - Sigma_R + 1e-10j)
    
    # Pre-compute time-frequency correlation matrices [Shape: (num_steps, N_w)]
    wt = t_grid[:, None] * w_grid[None, :]
    cos_wt = jnp.cos(wt)
    sin_wt = jnp.sin(wt)
    
    # Time-domain memory kernel: \chi^R(t) via Riemann integration
    exp_iwt = cos_wt - 1j * sin_wt
    chi_R_t = jnp.real(jnp.dot(exp_iwt, chi_R_w) * dw / (2.0 * jnp.pi))
    
    # Scale for the Heun solver 
    gamma_kernel = ((g**2) / n_spins) * chi_R_t
    
    # 4. Keldysh Noise Power Spectrum S(w)
    coth_w = jnp.where(abs_w > 1e-10, 1.0 / jnp.tanh(abs_w / (2.0 * T + 1e-12)), 0.0)
    S_w = -jnp.imag(chi_R_w) * coth_w
    S_w = jnp.where(S_w < 0, 0.0, S_w) # Positivity enforcement
    
    return gamma_kernel, S_w, chi_R_t, dw, cos_wt, sin_wt


@jax.jit(static_argnames=['num_steps'])
def generate_colored_noise(key, num_steps, dt, S_w, chi_R_t, dw, cos_wt, sin_wt, omega_0, g, n_photons_initial, n_spins):
    """
    Generates stochastic noise trajectories via Spectral Representation (Riemann Sum).
    Uses the pre-computed cos/sin matrices to reduce the integral to a single dot product.
    """
    N_w = S_w.shape[0]
    k_init_re, k_init_im, k_re, k_im = jax.random.split(key, 4)
    
    # 1. Generate Classical Colored Noise 
    A_k = jax.random.normal(k_re, shape=(N_w,))
    B_k = jax.random.normal(k_im, shape=(N_w,))
    
    # Thermodynamic variance weighting for the spectral sum
    amp = jnp.sqrt(S_w * dw / (2.0 * jnp.pi))
    
    # Fast Riemann summation via matrix multiplication
    bath_noise_t = jnp.dot(cos_wt, A_k * amp) + jnp.dot(sin_wt, B_k * amp)
    
    # 2. Coherent State Transient + Wigner Vacuum
    mean_field = jnp.sqrt(n_photons_initial) 
    vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
    vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
    alpha_0 = mean_field + (vacuum_fluc_re + 1j * vacuum_fluc_im)
    
    t_grid = jnp.arange(num_steps) * dt
    norm_factor = jnp.where(jnp.abs(chi_R_t[0]) > 1e-10, chi_R_t[0], 1.0)
    transient_alpha = alpha_0 * jnp.exp(-1j * omega_0 * t_grid) * (chi_R_t / norm_factor)
    phi_transient = 2.0 * jnp.real(transient_alpha)
    
    # 3. Total Cavity Noise Field
    phi_total = phi_transient + bath_noise_t
    
    # Effective magnetic noise kicking the spins
    xi_x = 2.0 * (g / jnp.sqrt(n_spins)) * phi_total
    
    return jnp.zeros((num_steps, 3)).at[:, 0].set(xi_x)


@jax.jit
def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, noise_traj, B_field, dt):
    """
    Computes the effective non-Markovian field.
    """
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices
    
    valid_mask = lag_indices > 0
    safe_lag = jnp.where(valid_mask, lag_indices, 0)
    gamma_causal = jnp.where(valid_mask, gamma_kernel[safe_lag], 0.0)

    # Memory integration (Convolution)
    memory_x = 0.5 * (jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    xi_field = 0.5 * noise_traj[step_idx, 0]
    
    eff_field_x = 0.5 * B_field[0] + xi_field - memory_x
    
    return jnp.array([eff_field_x, 0.5 * B_field[1], 0.5 * B_field[2]])


@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    """
    Predictor-Corrector step for the integro-differential equation.
    """
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]

    # --- PREDICTOR ---
    B_eff_p = compute_effective_field(S_curr, state_trajectory, curr_idx, gamma_kernel, noise_traj, B_field, dt)
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
    
    # --- CORRECTOR ---
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    B_eff_c = compute_effective_field(S_pred, traj_with_pred, step_idx, gamma_kernel, noise_traj, B_field, dt)
    
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
              
    return state_trajectory.at[step_idx].set(S_next), S_next


def run_integrated_twa_bundle(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1, w_max=20.0, N_w=2000):
    """
    Main execution bundle. Evaluates the Riemann matrices globally exactly once,
    then leverages vmap to multiply them against the local noise seeds.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # Pre-compute the continuous bath arrays and the massive cos/sin matrices once
    gamma_kernel_fine, S_w, chi_R_t, dw, cos_wt, sin_wt = compute_non_markovian_bath_functions(
        num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins, w_max, N_w)
    
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction, n_spins)
        
        # Inject the pre-computed matrices into the noise generator
        noise_traj = generate_colored_noise(
            k_noise, num_steps, dt, S_w, chi_R_t, dw, cos_wt, sin_wt, omega_0, g, n_photons_initial, n_spins)
        
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        def scan_body(carry, idx):
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, B_field, dt)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_trajs, axis=0)

    total_sum = jnp.zeros((num_steps, 3))
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Starting Riemann DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="Integrated DTWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        total_sum += process_batch_sum(current_keys)
        
    return total_sum / n_total