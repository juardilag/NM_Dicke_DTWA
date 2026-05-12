import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps', 'N_w'])
def compute_non_markovian_bath_functions(num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins, w_max=20.0, N_w=5000):
    t_grid = jnp.arange(num_steps) * dt
    w_grid = jnp.linspace(-w_max, w_max, N_w)
    dw = w_grid[1] - w_grid[0]
    
    # 1. J(w) definition: Antisymmetric spectral density [cite: 73, 77]
    abs_w = jnp.abs(w_grid)
    J_w = jnp.where(w_grid != 0, jnp.sign(w_grid) * alpha * omega_c * (abs_w/omega_c)**s * jnp.exp(-abs_w/omega_c), 0.0)
    
    # 2. MANUAL KRAMERS-KRONIG (Riemann Sum)
    # Calculates Re[Sigma] = (1/pi) * P.V. integral [J(w')/(w - w')] dw'
    # We use a vectorized grid to avoid the singularity at w = w'
    w_diff = w_grid[:, None] - w_grid[None, :]
    # Mask diagonal to avoid division by zero
    mask = jnp.eye(N_w)
    pv_kernel = (1.0 - mask) / (w_diff + mask) 
    Sigma_Real = jnp.dot(pv_kernel, J_w) * dw / jnp.pi
    
    # 3. Full chi^R(w) calculation [cite: 31]
    Sigma_R = Sigma_Real - 1j * jnp.pi * J_w 
    chi_R_w = 1.0 / (w_grid - omega_0 - Sigma_R + 1e-10j) - 1.0 / (w_grid + omega_0 - Sigma_R + 1e-10j)
    
    # 4. Time-frequency matrices (Your Original Method)
    wt = t_grid[:, None] * w_grid[None, :]
    cos_wt, sin_wt = jnp.cos(wt), jnp.sin(wt)
    # Manual inverse Fourier transform via dot product [cite: 67, 77]
    chi_R_t = jnp.real(jnp.dot(cos_wt - 1j*sin_wt, chi_R_w) * dw / (2.0 * jnp.pi))
    
    # 5. Kernel Scaling (Original 4.0 factor) [cite: 113]
    gamma_kernel = 4.0 * ((g**2) / n_spins) * chi_R_t
    
    # 6. Fluctuation-Dissipation Theorem for Noise Power S(w) [cite: 62]
    S_w = -jnp.imag(chi_R_w) * jnp.where(abs_w > 1e-10, 1.0/jnp.tanh(abs_w/(2.0*T + 1e-12)), 0.0)
    
    return gamma_kernel, jnp.where(S_w < 0, 0, S_w), chi_R_t, dw, cos_wt, sin_wt

@jax.jit(static_argnames=['num_steps', 'use_noise'])
def generate_colored_noise(key, num_steps, dt, S_w, chi_R_t, dw, cos_wt, sin_wt, omega_0, g, n_photons_initial, n_spins, use_noise=True):
    k_init_re, k_init_im, k_re, k_im = jax.random.split(key, 4)
    
    # Noise amp from Power Spectrum [cite: 126]
    amp = jnp.sqrt(S_w * dw / (2.0 * jnp.pi))
    
    bath_noise_t = jnp.where(use_noise, 
                             jnp.dot(cos_wt, jax.random.normal(k_re, S_w.shape) * amp) + \
                             jnp.dot(sin_wt, jax.random.normal(k_im, S_w.shape) * amp),
                             0.0)
    
    # Initial Wigner state of the Boson (Cavity) [cite: 124]
    alpha_noise = jax.random.normal(k_init_re)*jnp.sqrt(0.5) + 1j*jax.random.normal(k_init_im)*jnp.sqrt(0.5)
    alpha_0 = jnp.sqrt(n_photons_initial) + jnp.where(use_noise, alpha_noise, 0.0j)
    
    t_grid = jnp.arange(num_steps) * dt
    norm = jnp.where(jnp.abs(chi_R_t[0]) > 1e-10, chi_R_t[0], 1.0)
    
    # Transient decay [cite: 67]
    phi_transient = 2.0 * jnp.real(alpha_0 * jnp.exp(-1j*omega_0*t_grid) * (chi_R_t/norm))
    phi_total = phi_transient + bath_noise_t
    
    xi_x = -(2.0 * g / jnp.sqrt(n_spins)) * phi_total
    return jnp.zeros((num_steps, 3)).at[:, 0].set(xi_x)

@jax.jit
def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, noise_traj, B_field, dt):
    indices = jnp.arange(history_array.shape[0])
    lag = step_idx - indices
    mask = lag > 0
    gamma_causal = jnp.where(mask, gamma_kernel[jnp.where(mask, lag, 0)], 0.0)

    # Your original 0.5 scaling
    memory_x = 0.5 * (jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    xi_field = 0.5 * noise_traj[step_idx, 0]
    
    eff_x = 0.5 * B_field[0] + xi_field + memory_x 
    return jnp.array([eff_x, 0.5 * B_field[1], 0.5 * B_field[2]])

@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]

    # Predictor
    B_p = compute_effective_field(S_curr, state_trajectory, curr_idx, gamma_kernel, noise_traj, B_field, dt)
    mag_p = jnp.linalg.norm(B_p) + 1e-16
    ang_p = 2.0 * mag_p * dt # Sticking to your original 2.0 factor
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
                              batch_size=10_000, n_spins=1, w_max=20.0, N_w=5000, 
                              use_noise=True, use_sampling=True):
    """
    Modified bundle to allow Mean-Field (use_noise=False, use_sampling=False) 
    or TWA (use_noise=True, use_sampling=True) simulations.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    gamma_kernel_fine, S_w, chi_R_t, dw, cos_wt, sin_wt = compute_non_markovian_bath_functions(
        num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins, w_max, N_w)
    
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        # Initial condition: sampled from Wigner [cite: 137] or deterministic classical vector
        s0_sampled = discrete_spin_sampling_factorized(k_samp, initial_direction, n_spins) / 2.0
        s0_mean = (initial_direction * n_spins) / 2.0
        s0 = jnp.where(use_sampling, s0_sampled, s0_mean)
        
        # Generate noise trajectory (will be deterministic if use_noise=False)
        noise_traj = generate_colored_noise(
            k_noise, num_steps, dt, S_w, chi_R_t, dw, cos_wt, sin_wt, omega_0, g, n_photons_initial, n_spins, use_noise=use_noise)
        
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
    print(f"Starting Riemann {mode_name}: {n_total} trajectories in {n_batches} batches.")
    
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

@jax.jit
def extract_stationary_correlation(C_matrix):
    """
    Extracts C(tau) by aligning diagonals into columns and averaging.
    This avoids the Tracer error by keeping shapes static.
    """
    n = C_matrix.shape[0]
    
    # 1. Create an index grid for rows
    rows = jnp.arange(n)
    
    # 2. Roll each row i by -i positions. 
    # This moves the k-th diagonal element C[i, i+k] to the k-th column.
    def roll_row(i, row):
        return jnp.roll(row, -i)
    
    aligned_matrix = jax.vmap(roll_row)(rows, C_matrix)
    
    # 3. The first n//2 columns now contain the positive-lag diagonals.
    # We take the mean over the rows for each column.
    # Note: As lag increases, the number of valid (non-wrapped) elements decreases.
    # For a simple stationary approximation, we take the mean of the first n//2 columns.
    c_tau = jnp.mean(aligned_matrix[:, :n//2], axis=0)
    
    return c_tau

@jax.jit(static_argnames=['N_w'])
def fourier_transform_correlation(c_tau, t_grid, w_max=2.5, N_w=5000):
    """
    Performs a manual Riemann Fourier Transform from time (tau) to frequency (omega).
    S(w) = 2 * Real [ integral_0^inf C(tau) * exp(i*w*tau) dtau ]
    """
    dt = t_grid[1] - t_grid[0]
    tau_grid = jnp.arange(len(c_tau)) * dt
    w_grid = jnp.linspace(0, w_max, N_w)
    
    # Time-Frequency matrix for the transform
    # Using the same Riemann structure as your compute_non_markovian_bath_functions
    w_tau = w_grid[:, None] * tau_grid[None, :]
    kernel = jnp.cos(w_tau) + 1j * jnp.sin(w_tau)
    
    # S(w) is the power spectrum of the fluctuations
    S_w = 2.0 * jnp.real(jnp.dot(kernel, c_tau) * dt)
    
    return w_grid, S_w

