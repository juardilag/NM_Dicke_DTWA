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
    t_grid = jnp.arange(num_steps) * dt
    w_grid = jnp.linspace(-w_max, w_max, N_w)
    dw = w_grid[1] - w_grid[0]
    
    # J(w) definition
    abs_w = jnp.abs(w_grid)
    J_w = jnp.where(w_grid != 0, jnp.sign(w_grid) * alpha * omega_c * (abs_w/omega_c)**s * jnp.exp(-abs_w/omega_c), 0.0)
    
    # chi^R(w)
    Sigma_R = -1j * jnp.pi * J_w 
    chi_R_w = 1.0 / (w_grid - omega_0 - Sigma_R + 1e-10j) - 1.0 / (w_grid + omega_0 - Sigma_R + 1e-10j)
    
    # Time-frequency matrices
    wt = t_grid[:, None] * w_grid[None, :]
    cos_wt, sin_wt = jnp.cos(wt), jnp.sin(wt)
    chi_R_t = jnp.real(jnp.dot(cos_wt - 1j*sin_wt, chi_R_w) * dw / (2.0 * jnp.pi))
    
    # FACTOR FIX: Matches your compute_memory_kernel logic
    gamma_kernel = 4.0 * ((g**2) / n_spins) * chi_R_t
    
    S_w = -jnp.imag(chi_R_w) * jnp.where(abs_w > 1e-10, 1.0/jnp.tanh(abs_w/(2.0*T + 1e-12)), 0.0)
    
    return gamma_kernel, jnp.where(S_w < 0, 0, S_w), chi_R_t, dw, cos_wt, sin_wt

@jax.jit(static_argnames=['num_steps'])
def generate_colored_noise(key, num_steps, dt, S_w, chi_R_t, dw, cos_wt, sin_wt, omega_0, g, n_photons_initial, n_spins):
    k_init_re, k_init_im, k_re, k_im = jax.random.split(key, 4)
    
    # 1. Classical Noise from Bath
    amp = jnp.sqrt(S_w * dw / (2.0 * jnp.pi))
    bath_noise_t = jnp.dot(cos_wt, jax.random.normal(k_re, S_w.shape) * amp) + \
                   jnp.dot(sin_wt, jax.random.normal(k_im, S_w.shape) * amp)
    
    # 2. Transient from Initial Cavity State
    alpha_0 = jnp.sqrt(n_photons_initial) + (jax.random.normal(k_init_re)*jnp.sqrt(0.5) + 1j*jax.random.normal(k_init_im)*jnp.sqrt(0.5))
    t_grid = jnp.arange(num_steps) * dt
    norm = jnp.where(jnp.abs(chi_R_t[0]) > 1e-10, chi_R_t[0], 1.0)
    
    # phi = 2*Re[alpha]
    phi_transient = 2.0 * jnp.real(alpha_0 * jnp.exp(-1j*omega_0*t_grid) * (chi_R_t/norm))
    phi_total = phi_transient + bath_noise_t
    
    # DEBUG FIX: The effective field carries a MINUS sign (B = -dH/dJ)
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
    eff_x = 0.5 * B_field[0] + xi_field - memory_x 
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


def run_integrated_twa_bundle(keys, t_grid, omega_0, alpha, omega_c, s, T, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1, w_max=20.0, N_w=2000):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    gamma_kernel_fine, S_w, chi_R_t, dw, cos_wt, sin_wt = compute_non_markovian_bath_functions(
        num_steps, dt, omega_0, alpha, omega_c, s, T, g, n_spins, w_max, N_w)
    
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction, n_spins) / 2.0
        
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