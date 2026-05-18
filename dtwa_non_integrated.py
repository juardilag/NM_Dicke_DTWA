import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized, cavity_wigner_sampling
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_non_markovian_bath_kernels(num_steps, dt, alpha, omega_c, s, T, w_max=20.0, N_w=5000):
    """
    Computes the time-domain retarded self-energy Sigma^R(t) and the 
    noise power spectrum S(w) directly from the customized spectral density J(w).
    """
    t_grid = jnp.arange(num_steps) * dt
    w_grid_pos = jnp.linspace(1e-10, w_max, N_w)
    dw = w_grid_pos[1] - w_grid_pos[0]
    
    # Define continuous one-sided spectral density J(w) for positive frequencies
    J_w_pos = alpha * omega_c * (w_grid_pos / omega_c)**s * jnp.exp(-w_grid_pos / omega_c)
    
    # Compute Sigma^R(t) via continuous numerical half-sided Fourier Transform
    # Sigma^R(t) = -i * \int_0^\infty dw J(w) e^{-iwt}
    wt = t_grid[:, None] * w_grid_pos[None, :]
    fourier_integral = jnp.dot(jnp.cos(wt) - 1j * jnp.sin(wt), J_w_pos) * dw
    Sigma_R_t = -1j * fourier_integral
    
    # Noise power allocation via the Fluctuation-Dissipation Theorem
    S_w_pos = J_w_pos * (1.0 / jnp.tanh(w_grid_pos / (2.0 * T + 1e-12)))
    
    return Sigma_R_t, S_w_pos, w_grid_pos, dw


@jax.jit(static_argnames=['num_steps'])
def generate_colored_noise_trajectory(key, num_steps, dt, S_w_pos, w_grid_pos, dw, use_noise=True):
    """
    Generates a complex colored noise trajectory xi(t) acting directly on the cavity.
    """
    half_N = S_w_pos.shape[0]
    k_re, k_im = jax.random.split(key)
    
    # Amplitude scaling factor based on the spectral power per frequency slice
    amp = jnp.sqrt(S_w_pos * dw)
    
    noise_re = jax.random.normal(k_re, (half_N,))
    noise_im = jax.random.normal(k_im, (half_N,))
    complex_seed = (noise_re + 1j * noise_im) / jnp.sqrt(2.0)
    
    t_grid = jnp.arange(num_steps) * dt
    wt = t_grid[:, None] * w_grid_pos[None, :]
    
    # Transform back to the time-domain to construct the stochastic driving history
    xi_t = jnp.where(use_noise, 
                     jnp.dot(jnp.cos(wt) - 1j * jnp.sin(wt), complex_seed * amp), 
                     0.0 + 0j)
    return xi_t


@jax.jit(static_argnames=['num_steps'])
def non_markovian_coupled_heun_step(S_history, psi_history, step_idx, noise_traj, Sigma_R_t, B_field_val, g, n_spins, dt, num_steps):
    """
    Performs a non-Markovian Predictor-Corrector step utilizing a masked vector dot product.
    """
    curr_idx = step_idx - 1
    S_curr = S_history[curr_idx]
    psi_curr = psi_history[curr_idx]
    
    # Physical coupling scaling: g_eff = 2*sqrt(2)*g / sqrt(N)
    g_eff = 2.0 * jnp.sqrt(2.0) * g / jnp.sqrt(n_spins)
    
    # --- 1. PREDICTOR STEP ---
    # Causal masking to calculate the predictor memory integral at t = (step_idx - 1) * dt
    mask_p = jnp.arange(num_steps) < step_idx
    lag_p = curr_idx - jnp.arange(num_steps)
    kernel_p = jnp.where(mask_p, Sigma_R_t[jnp.maximum(0, lag_p)], 0.0 + 0j)
    memory_p = jnp.dot(kernel_p, psi_history) * dt
    
    # Spin effective magnetic field matching standard Larmor frequency scaling
    B_eff_p = B_field_val + jnp.array([g_eff * 2.0 * jnp.real(psi_curr), 0.0, 0.0])
    mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    ang_p = 2.0 * mag_p * dt  # Maintaining your original 2.0 spin scaling factor
    axis_p = B_eff_p / mag_p
    
    S_pred = (S_curr * jnp.cos(ang_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(ang_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(ang_p)))
    
    # Cavity non-Markovian Langevin derivative equation
    dpsi_p = -1j * memory_p - 1j * g_eff * S_curr[0] + 1j * noise_traj[curr_idx]
    psi_pred = psi_curr + dpsi_p * dt
    
    # Temporary allocation arrays to supply the corrector step with a guess of the future
    psi_history_pred = psi_history.at[step_idx].set(psi_pred)
    
    # --- 2. CORRECTOR STEP ---
    # Memory integral at t = step_idx * dt using the predicted future value
    mask_c = jnp.arange(num_steps) <= step_idx
    lag_c = step_idx - jnp.arange(num_steps)
    kernel_c = jnp.where(mask_c, Sigma_R_t[jnp.maximum(0, lag_c)], 0.0 + 0j)
    memory_c = jnp.dot(kernel_c, psi_history_pred) * dt
    
    B_eff_c = B_field_val + jnp.array([g_eff * 2.0 * jnp.real(psi_pred), 0.0, 0.0])
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    mag_c = jnp.linalg.norm(B_eff_avg) + 1e-16
    ang_c = 2.0 * mag_c * dt
    axis_c = B_eff_avg / mag_c
    
    S_next = (S_curr * jnp.cos(ang_c) + 
              jnp.cross(axis_c, S_curr) * jnp.sin(ang_c) + 
              axis_c * jnp.dot(axis_c, S_curr) * (1.0 - jnp.cos(ang_c)))
    
    dpsi_c = -1j * memory_c - 1j * g_eff * S_pred[0] + 1j * noise_traj[step_idx]
    psi_next = psi_curr + 0.5 * (dpsi_p + dpsi_c) * dt
    
    return S_history.at[step_idx].set(S_next), psi_history.at[step_idx].set(psi_next)


def run_coupled_non_markovian_twa_bundle(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, 
                                         batch_size=1000, n_spins=1, w_max=20.0, N_w=5000, use_noise=True, use_sampling=True):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # Pre-calculate environmental structures based on selected spectral density
    Sigma_R_t, S_w_pos, w_grid_pos, dw = compute_non_markovian_bath_kernels(
        num_steps, dt, alpha, omega_c, s, T, w_max, N_w)

    def solve_single_trajectory(key):
        k_samp_spin, k_samp_cav, k_noise = jax.random.split(key, 3)
        
        # Sample Initial Conditions
        s0_sampled = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins) / 2.0
        s0_mean = (initial_direction * n_spins) / 2.0
        s0 = jnp.where(use_sampling, s0_sampled, s0_mean)
        
        psi0_sampled = cavity_wigner_sampling(k_samp_cav, n_photons_initial)
        psi0_mean = jnp.sqrt(jnp.array(n_photons_initial, dtype=jnp.float64)) + 0j
        psi0 = jnp.where(use_sampling, psi0_sampled, psi0_mean)
        
        # Allocate explicit time series buffers
        S_history = jnp.zeros((num_steps, 3)).at[0].set(s0)
        psi_history = jnp.zeros((num_steps,), dtype=jnp.complex128).at[0].set(psi0)
        
        # Generate custom colored noise profiles
        noise_traj = generate_colored_noise_trajectory(
            k_noise, num_steps, dt, S_w_pos, w_grid_pos, dw, use_noise=use_noise)
        
        def scan_body(carry, step_idx):
            S_hist, psi_hist = carry
            # Shift the bare frequency component omega_0 into the field directly
            B_val = B_field[step_idx - 1].at[2].add(omega_0) 
            
            S_next_hist, psi_next_hist = non_markovian_coupled_heun_step(
                S_hist, psi_hist, step_idx, noise_traj, Sigma_R_t, B_val, g, n_spins, dt, num_steps)
            
            return (S_next_hist, psi_next_hist), None

        init_carry = (S_history, psi_history)
        (final_S_history, final_psi_history), _ = jax.lax.scan(scan_body, init_carry, jnp.arange(1, num_steps))
        
        return final_S_history, final_psi_history

    @jax.jit
    def process_batch(batch_keys):
        return jax.vmap(solve_single_trajectory)(batch_keys)

    all_S, all_psi = [], []
    n_batches = int(jnp.ceil(n_total / batch_size))
    mode_name = "TWA" if use_noise else "Mean-Field"
    
    print(f"Starting Coupled Non-Markovian {mode_name}: {n_total} trajectories across {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc=f"Non-Markovian Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_S, batch_psi = process_batch(current_keys)
        all_S.append(batch_S)
        all_psi.append(batch_psi)
        
    return jnp.concatenate(all_S, axis=0), jnp.concatenate(all_psi, axis=0)