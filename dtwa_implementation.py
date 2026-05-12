import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps', 'N_fft'])
def compute_non_markovian_bath_functions(num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins, w_max=50.0, N_fft=16384):
    t_sim_grid = jnp.arange(num_steps) * dt
    w_grid = jnp.linspace(0, w_max, N_fft // 2 + 1)
    dw = w_grid[1] - w_grid[0]
    
    # J(w) definition [cite: 73]
    J_w = alpha * omega_c * (w_grid/omega_c)**s * jnp.exp(-w_grid/omega_c)
    
    # chi^R(w) calculation [cite: 31, 77]
    Sigma_R = -1j * jnp.pi * J_w 
    chi_R_w = 1.0 / (w_grid - omega_0 - Sigma_R + 1e-10j) - 1.0 / (w_grid + omega_0 - Sigma_R + 1e-10j)
    
    # Robust IFFT 
    chi_R_t_full = jnp.fft.irfft(chi_R_w, n=N_fft) * (N_fft * dw / jnp.pi)
    
    # --- FIX: Map FFT time-steps to Simulation time-steps ---
    t_fft_grid = jnp.arange(N_fft) * (jnp.pi / w_max)
    chi_R_t = jnp.interp(t_sim_grid, t_fft_grid, chi_R_t_full)
    # -------------------------------------------------------
    
    gamma_kernel = 4.0 * ((g**2) / n_spins) * chi_R_t 
    S_w = -jnp.imag(chi_R_w) * jnp.where(w_grid > 1e-10, 1.0/jnp.tanh(w_grid/(2.0*T + 1e-12)), 0.0) 
    
    return gamma_kernel, S_w, chi_R_t, dw, w_grid

@jax.jit(static_argnames=['num_steps', 'use_noise'])
def generate_colored_noise(key, num_steps, dt, S_w, w_grid, dw, omega_0, g, n_photons_initial, n_spins, use_noise=True):
    k_init, k_noise = jax.random.split(key)
    N_fft_noise = (len(S_w) - 1) * 2
    t_sim_grid = jnp.arange(num_steps) * dt
    
    # Frequency Domain Noise Construction [cite: 124]
    rand_re = jax.random.normal(k_noise, (len(S_w),))
    rand_im = jax.random.normal(k_noise + 1, (len(S_w),))
    noise_w = (rand_re + 1j*rand_im) * jnp.sqrt(S_w * dw / (2.0 * jnp.pi))
    
    # Transform to Time Domain
    bath_noise_full = jnp.fft.irfft(noise_w, n=N_fft_noise) * N_fft_noise
    
    # --- FIX: Interpolate noise to simulation grid ---
    t_fft_noise = jnp.arange(N_fft_noise) * (jnp.pi / jnp.max(w_grid))
    bath_noise_t = jnp.interp(t_sim_grid, t_fft_noise, bath_noise_full)
    bath_noise_t = jnp.where(use_noise, bath_noise_t, 0.0)
    # -------------------------------------------------
    
    alpha_0 = jnp.sqrt(n_photons_initial)
    phi_transient = 2.0 * jnp.real(alpha_0 * jnp.exp(-1j*omega_0*t_sim_grid)) 
    
    phi_total = phi_transient + bath_noise_t
    xi_x = -(2.0 * g / jnp.sqrt(n_spins)) * phi_total
    
    return jnp.zeros((num_steps, 3)).at[:, 0].set(xi_x)

@jax.jit
def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, noise_traj, B_field, dt):
    indices = jnp.arange(history_array.shape[0])
    lag = step_idx - indices
    mask = lag > 0
    gamma_causal = jnp.where(mask, gamma_kernel[jnp.where(mask, lag, 0)], 0.0)

    # Memory integration
    # DEBUG FIX: The memory kernel contribution should be ADDED to the field
    memory_x = 0.5 * (jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    xi_field = 0.5 * noise_traj[step_idx, 0]
    
    # The field is B_bare + Noise + Memory
    eff_x = 0.5 * B_field[0] + xi_field + memory_x 
    return jnp.array([eff_x, 0.5 * B_field[1], 0.5 * B_field[2]])


@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]

    # Predictor
    B_p = compute_effective_field(S_curr, state_trajectory, curr_idx, gamma_kernel, noise_traj, B_field, dt)
    # FACTOR FIX: Restored your 2.0 angle factor
    mag_p = jnp.linalg.norm(B_p) + 1e-16
    ang_p = 2.0 * mag_p * dt 
    axis_p = B_p / mag_p
    S_pred = (S_curr * jnp.cos(ang_p) + jnp.cross(axis_p, S_curr) * jnp.sin(ang_p) + axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(ang_p)))
    
    # Corrector
    B_c = compute_effective_field(S_pred, state_trajectory.at[step_idx].set(S_pred), step_idx, gamma_kernel, noise_traj, B_field, dt)
    B_avg = 0.5 * (B_p + B_c)
    mag_a = jnp.linalg.norm(B_avg) + 1e-16
    ang_a = 2.0 * mag_a * dt
    axis_a = B_avg / mag_a
    S_next = (S_curr * jnp.cos(ang_a) + jnp.cross(axis_a, S_curr) * jnp.sin(ang_a) + axis_a * jnp.dot(axis_a, S_curr) * (1.0 - jnp.cos(ang_a)))
              
    return state_trajectory.at[step_idx].set(S_next), S_next


def run_integrated_twa_bundle(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, 
                              batch_size=10_000, n_spins=1, 
                              use_noise=True, use_sampling=True):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # UPDATED: Returns w_grid instead of sin/cos matrices
    gamma_kernel_fine, S_w, chi_R_t, dw, w_grid = compute_non_markovian_bath_functions(
        num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins)
    
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        # Initial condition: sampled from Wigner [cite: 137] or deterministic vector
        s0_sampled = discrete_spin_sampling_factorized(k_samp, initial_direction, n_spins) / 2.0
        s0_mean = (initial_direction * n_spins) / 2.0
        s0 = jnp.where(use_sampling, s0_sampled, s0_mean)
        
        # UPDATED: Passes w_grid to the noise generator
        noise_traj = generate_colored_noise(
            k_noise, num_steps, dt, S_w, w_grid, dw, omega_0, g, n_photons_initial, n_spins, use_noise=use_noise)
        
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        def scan_body(carry, idx):
            current_B = B_field[idx] 
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, current_B, dt)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    @jax.jit
    def process_batch(batch_keys):
        return jax.vmap(solve_single_trajectory)(batch_keys)

    all_trajectories = []
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    mode_name = "TWA" if use_noise else "Mean-Field"
    print(f"Starting {mode_name}: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc=f"Integrated {mode_name} Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        all_trajectories.append(process_batch(current_keys))
        
    return jnp.concatenate(all_trajectories, axis=0)

def calculate_correlation(sx_trajectories):
    """
    sx_trajectories: Array of shape (n_trajectories, n_timesteps)
    Returns: 2D array C(t, t')
    """
    n_traj, n_steps = sx_trajectories.shape
    # Subtract the mean to get fluctuations: δSx = Sx - <Sx>
    mean_sx = jnp.mean(sx_trajectories, axis=0)
    fluctuations = sx_trajectories - mean_sx
    
    # Compute the average over trajectories
    # C[t, t'] = (1/N) * sum( δSx(t) * δSx(t') )
    C = (fluctuations.T @ fluctuations) / n_traj
    return C

def calculate_response(dt, t_prime_idx, perturbed_mean_sx, reference_mean_sx, h_strength):
    """
    Calculates χ(t, t') = δ<Sx(t)> / δh(t')
    """
    # Difference in the average evolution
    delta_sx = perturbed_mean_sx - reference_mean_sx
    
    # Response is the change divided by the perturbation strength
    chi = delta_sx / (h_strength * dt)
    
    # Ensure causality: chi = 0 for t < t' using JAX's immutable update syntax
    chi = chi.at[:t_prime_idx].set(0.0)
    
    return chi