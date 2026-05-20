from dtwa_non_integrated import solve_single_trajectory, compute_explicit_bath_kernels
import numpy as np
import jax
import jax.numpy as jnp
import jax.debug
from tqdm import tqdm

@jax.jit
def _accumulate_correlation_components(spin_ensemble, cavity_ensemble, n_spins):
    """
    Computes batched inner-product accumulations on the device.
    Prevents allocating massive uncollapsed global matrices.
    """
    j_val = n_spins / 2.0
    
    sx = spin_ensemble[:, :, 0] / j_val
    alpha = cavity_ensemble
    
    sum_sx = jnp.sum(sx, axis=0)
    sum_alpha = jnp.sum(alpha, axis=0)
    
    outer_sx = sx.T @ sx
    outer_alpha = jnp.real(alpha.T @ jnp.conj(alpha))
    
    return sum_sx, sum_alpha, outer_sx, outer_alpha


def calculate_correlations(keys, t_grid, p, B_field, w_max=20.0, N_w=5000):
    """
    Runs TWA trajectories in isolated chunks, building global correlation matrices
    on-the-fly. Cavity external electric fields default safely to zero.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    batch_size = p['batch_size']
    j_val = p['n_spins'] / 2.0
    
    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(
        num_steps, dt, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], w_max, N_w
    )

    # Initialize cavity_drive to zero for the unperturbed correlation calculations
    cavity_drive = jnp.zeros(num_steps, dtype=jnp.float64)

    n_batches = int(jnp.ceil(n_total / batch_size))
    pbar = tqdm(total=n_batches, desc=f"Running DTWA of {n_total} trajectories in {n_batches} batches")

    def update_pbar():
        pbar.update(1)

    @jax.jit
    def process_batch(batch_keys):
        batch_S, batch_alpha = jax.vmap(lambda k: solve_single_trajectory(
            k, t_grid, p['omega_0'], B_field, p['g'], p['n_photons_initial'], p['initial_direction'], 
            p['n_spins'], dt, num_steps, Sigma_R_t, S_bath_w, w_grid, dw,
            True, True, cavity_drive
        ))(batch_keys)
        
        components = _accumulate_correlation_components(batch_S, batch_alpha, p['n_spins'])
        jax.debug.callback(update_pbar)
        return components

    # Allocate clean accumulators on Host CPU RAM
    global_sum_sx = np.zeros(num_steps, dtype=np.float64)
    global_sum_alpha = np.zeros(num_steps, dtype=np.complex128)
    global_outer_sx = np.zeros((num_steps, num_steps), dtype=np.float64)
    global_outer_alpha = np.zeros((num_steps, num_steps), dtype=np.float64)

    try:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_total)
            
            sum_sx, sum_alpha, out_sx, out_alpha = process_batch(keys[start_idx:end_idx])
            
            # Synchronize hardware threads before moving data blocks
            sum_sx.block_until_ready()
            
            global_sum_sx += np.array(sum_sx)
            global_sum_alpha += np.array(sum_alpha)
            global_outer_sx += np.array(out_sx)
            global_outer_alpha += np.array(out_alpha)
    finally:
        pbar.close()

    # Apply covariance expansion: <ΔA ΔB> = <AB> - <A><B>
    mean_sx = global_sum_sx / n_total
    mean_alpha = global_sum_alpha / n_total
    
    C_spin = (global_outer_sx / n_total) - np.outer(mean_sx, mean_sx)
    C_cavity = ((global_outer_alpha / n_total) - np.real(np.outer(mean_alpha, np.conj(mean_alpha)))) / j_val

    jax.clear_caches()
    return C_spin, C_cavity

@jax.jit
def extract_stationary_correlation(C_matrix):
    """Computes time-averaged stationary correlation vectors along lags."""
    n = C_matrix.shape[0]
    rows = jnp.arange(n, dtype=jnp.int64)
    
    def roll_row(i, row):
        return jnp.roll(row, -i)
    aligned_matrix = jax.vmap(roll_row)(rows, C_matrix)
    
    lags = jnp.arange(n, dtype=jnp.int64)
    counts = n - lags
    c_tau_sum = jnp.sum(aligned_matrix * (rows[:, None] + lags[None, :] < n), axis=0)
    
    return c_tau_sum / jnp.where(counts > 0, counts, 1.0)

@jax.jit
def fourier_transform_correlation(c_tau, dt, w_grid):
    """Fourier transform of correlation utilizing explicit float64 literals."""
    tau_grid = jnp.arange(len(c_tau), dtype=jnp.float64) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    weights = jnp.ones_like(c_tau).at[0].set(0.5).at[-1].set(0.5)
    
    return 2.0 * jnp.dot(jnp.cos(w_tau), c_tau * weights) * dt

def calculate_responses(keys, t_grid, p, t_pulse, epsilon=0.01, w_max=20.0, N_w=5000):
    """
    Measures the auto-susceptibility for BOTH the spin and cavity systems simultaneously.
    Optimized to run only 3 simulation sets instead of 4 by sharing a single Base state.
    
    Returns:
        response_spin:   The magnetic x-axis spin susceptibility response array [t_pulse:]
        response_cavity: The electric field cavity quadrature response array [t_pulse:]
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    batch_size = p['batch_size']
    pulse_idx = jnp.searchsorted(t_grid, t_pulse)
    j_val = p['n_spins'] / 2.0

    # -----------------------------------------------------------------
    # Define Perturbation Fields
    # -----------------------------------------------------------------
    # 1. Base Setup (Zero external drivers except baseline B_z)
    B_base = jnp.zeros((num_steps, 3), dtype=jnp.float64).at[:, 2].set(p['B_z'])
    E_base = jnp.zeros(num_steps, dtype=jnp.float64)

    # 2. Spin Perturbation Setup (Magnetic delta kick in B_x)
    B_pert_spin = B_base.at[pulse_idx, 0].add(epsilon / dt)
    E_pert_spin = E_base

    # 3. Cavity Perturbation Setup (Electric delta kick in E_field)
    B_pert_cav = B_base
    E_pert_cav = E_base.at[pulse_idx].add(epsilon / dt)

    # Compute common non-Markovian kernels
    Sigma_R_t, S_bath_w, w_grid, dw = compute_explicit_bath_kernels(
        num_steps, dt, p['omega_0'], p['alpha'], p['omega_c'], p['s'], p['T'], w_max, N_w
    )

    n_batches = int(jnp.ceil(n_total / batch_size))
    
    # Unified batch processing operator returning both physical properties
    @jax.jit
    def process_response_batch(batch_keys, B_field_val, E_field_val):
        batch_S, batch_alpha = jax.vmap(lambda k: solve_single_trajectory(
            k, t_grid, p['omega_0'], B_field_val, p['g'], p['n_photons_initial'], p['initial_direction'], 
            p['n_spins'], dt, num_steps, Sigma_R_t, S_bath_w, w_grid, dw, 
            use_noise=True, use_sampling=True, cavity_drive=E_field_val
        ))(batch_keys)
        
        # Return sums of both channels simultaneously
        return jnp.sum(batch_S[:, :, 0], axis=0), jnp.sum(jnp.real(batch_alpha), axis=0)

    # =====================================================================
    # STAGE 1: Run Shared Base Ensemble Simulation
    # =====================================================================
    pbar_base = tqdm(total=n_batches, desc="Simulation 1/3: Shared Base State")
    sum_base_spin = np.zeros(num_steps, dtype=np.float64)
    sum_base_cav  = np.zeros(num_steps, dtype=np.float64)
    
    for i in range(n_batches):
        current_keys = keys[i*batch_size : min((i+1)*batch_size, n_total)]
        res_spin, res_cav = process_response_batch(current_keys, B_base, E_base)
        res_spin.block_until_ready()
        
        sum_base_spin += np.array(res_spin)
        sum_base_cav  += np.array(res_cav)
        pbar_base.update(1)
    pbar_base.close()

    jax.clear_caches()

    # =====================================================================
    # STAGE 2: Run Perturbed Spin Ensemble Simulation
    # =====================================================================
    pbar_spin = tqdm(total=n_batches, desc="Simulation 2/3: Perturbed Spin (Magnetic)")
    sum_pert_spin = np.zeros(num_steps, dtype=np.float64)
    
    for i in range(n_batches):
        current_keys = keys[i*batch_size : min((i+1)*batch_size, n_total)]
        res_spin, _ = process_response_batch(current_keys, B_pert_spin, E_pert_spin)
        res_spin.block_until_ready()
        
        sum_pert_spin += np.array(res_spin)
        pbar_spin.update(1)
    pbar_spin.close()

    jax.clear_caches()

    # =====================================================================
    # STAGE 3: Run Perturbed Cavity Ensemble Simulation
    # =====================================================================
    pbar_cav = tqdm(total=n_batches, desc="Simulation 3/3: Perturbed Cavity (Electric)")
    sum_pert_cav = np.zeros(num_steps, dtype=np.float64)
    
    for i in range(n_batches):
        current_keys = keys[i*batch_size : min((i+1)*batch_size, n_total)]
        _, res_cav = process_response_batch(current_keys, B_pert_cav, E_pert_cav)
        res_cav.block_until_ready()
        
        sum_pert_cav += np.array(res_cav)
        pbar_cav.update(1)
    pbar_cav.close()

    # =====================================================================
    # Normalization, Scale Alignment & Slicing
    # =====================================================================
    # Spin Channel Scaling
    mean_base_spin = (sum_base_spin / n_total) / j_val
    mean_pert_spin = (sum_pert_spin / n_total) / j_val
    response_spin  = (mean_pert_spin - mean_base_spin) / (0.5 * epsilon)

    # Cavity Channel Scaling
    mean_base_cav = (sum_base_cav / n_total) / jnp.sqrt(j_val)
    mean_pert_cav = (sum_pert_cav / n_total) / jnp.sqrt(j_val)
    response_cavity = (mean_pert_cav - mean_base_cav) / epsilon

    jax.clear_caches()
    return response_spin[pulse_idx:], response_cavity[pulse_idx:]

@jax.jit
def fourier_transform_response(chi_tau, dt, w_grid):
    """
    Computes the imaginary component of the susceptibility via a Fourier transform.
    Utilizes clear 64-bit float types to secure your targeted precision profiles.
    """
    tau_grid = jnp.arange(len(chi_tau), dtype=jnp.float64) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    weights = jnp.ones_like(chi_tau).at[0].set(0.5).at[-1].set(0.5)
    
    imag_chi_w = jnp.dot(jnp.sin(w_tau), chi_tau * weights) * dt
    return imag_chi_w