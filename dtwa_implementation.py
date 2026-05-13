import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

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
    # Split the grid to use only positive frequencies for noise generation
    half_N = S_w.shape[0] // 2
    w_pos = jnp.linspace(0, 20.0, half_N) # Positive frequencies
    S_pos = S_w[half_N:] # S(w) is symmetric, take the positive side
    
    k_re, k_im = jax.random.split(key)
    
    # Standard formula for colored noise from S(w): 
    # Var = Integral[ S(w) dw / 2pi ]
    # We use sqrt(2) because we only integrate over positive frequencies
    amp = jnp.sqrt(S_pos * dw / jnp.pi) 
    
    # Regenerate time-frequency matrices for positive frequencies only
    t_grid = jnp.arange(num_steps) * dt
    wt_pos = t_grid[:, None] * w_pos[None, :]
    
    # Generate noise using independent Gaussian variables for phase and amplitude
    noise_re = jax.random.normal(k_re, (half_N,))
    noise_im = jax.random.normal(k_im, (half_N,))
    
    # This construction ensures the noise is real and matches the spectral density exactly
    bath_noise_t = jnp.where(use_noise, 
                             jnp.dot(jnp.cos(wt_pos), noise_re * amp) + \
                             jnp.dot(jnp.sin(wt_pos), noise_im * amp), 
                             0.0)
    
    # Based on Eq. 34: B_noise = - (1/sqrt(2)) * xi
    # Based on Eq. 126: <xi xi> = (8 g^2 / N) D_K
    # This matches your 2.0 * g / sqrt(n_spins) scaling perfectly [cite: 126, 138]
    xi_x = -(2.0 * g / jnp.sqrt(n_spins)) * bath_noise_t
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
    n = C_matrix.shape[0]
    rows = jnp.arange(n)
    
    def roll_row(i, row):
        return jnp.roll(row, -i)
    aligned_matrix = jax.vmap(roll_row)(rows, C_matrix)
    
    # Calculate number of samples for each lag to unbias the estimator
    lags = jnp.arange(n)
    counts = n - lags
    c_tau_sum = jnp.sum(aligned_matrix * (rows[:, None] + lags[None, :] < n), axis=0)
    
    # FIX: Unbiased estimator prevents artificial damping of the tail
    return c_tau_sum / jnp.where(counts > 0, counts, 1.0)

@jax.jit
def fourier_transform_correlation(c_tau, dt, w_grid):
    """
    S(w) = 2 * Integral[ C(tau) * cos(w*tau) ] d_tau
    """
    tau_grid = jnp.arange(len(c_tau)) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    
    # Trapezoidal weights for Riemann sum: 0.5 at the boundaries
    weights = jnp.ones_like(c_tau).at[0].set(0.5).at[-1].set(0.5)
    
    # Standard 2.0 factor for double-sided power spectrum of real symmetric C(t)
    S_w = 2.0 * jnp.dot(jnp.cos(w_tau), c_tau * weights) * dt
    return S_w

def measure_linear_response_fdt(keys, t_grid, p, t_pulse, epsilon=0.001):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    pulse_idx = jnp.searchsorted(t_grid, t_pulse)
    j_val = p['n_spins'] / 2.0

    # 1. Prepare Field Arrays
    B_base = jnp.zeros((num_steps, 3)).at[:, 2].set(p['B_z'])
    B_pert = B_base.at[pulse_idx, 0].add(epsilon / dt)

    print(f">>> Propagating Base Ensemble (t_pulse={t_pulse})...")
    res_base = run_integrated_twa_bundle(
        keys, t_grid, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], 
        B_base, p['g'], p['n_photons_initial'], p['initial_direction'],
        n_spins=p['n_spins']
    )
    
    print(f">>> Propagating Perturbed Ensemble (Same Keys)...")
    res_pert = run_integrated_twa_bundle(
        keys, t_grid, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], 
        B_pert, p['g'], p['n_photons_initial'], p['initial_direction'],
        n_spins=p['n_spins']
    )

    # 2. FIX: Divide by j_val to get intensive s_x = J_x / j
    mean_sx_base = jnp.mean(res_base[:, :, 0], axis=0) / j_val
    mean_sx_pert = jnp.mean(res_pert[:, :, 0], axis=0) / j_val
    
    # 3. FIX: Divide by physical field perturbation (0.5 * epsilon)
    # This gives us exactly chi_sB (intensive susceptibility)
    response = (mean_sx_pert - mean_sx_base) / epsilon

    return response[pulse_idx:]

@jax.jit
def fourier_transform_response(chi_tau, dt, w_grid):
    """
    Im[chi(w)] = Integral[ chi(tau) * sin(w*tau) ] d_tau
    """
    tau_grid = jnp.arange(len(chi_tau)) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    
    # Use identical Trapezoidal weights for consistency
    weights = jnp.ones_like(chi_tau).at[0].set(0.5).at[-1].set(0.5)
    
    # No 2.0 here: the 2.0 in the FDT relation comes from the S(w) definition above
    imag_chi_w = jnp.dot(jnp.sin(w_tau), chi_tau * weights) * dt
    return imag_chi_w