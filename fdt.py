from dtwa_non_integrated import run_coupled_non_markovian_twa_bundle
import jax.numpy as jnp
import jax

def calculate_spin_correlation(sx_trajectories, j_val):
    """
    Computes the intensive spin auto-correlation matrix.
    Expects a 2D array `sx_trajectories` of shape (n_trajectories, n_steps).
    """
    n_traj = sx_trajectories.shape[0]
    # Since sx_trajectories is already 2D, we scale it directly
    sx_intensive = sx_trajectories / j_val
    
    mean_sx = jnp.mean(sx_intensive, axis=0)
    fluctuations = sx_intensive - mean_sx
    
    # Intensive correlation matrix
    C = (fluctuations.T @ fluctuations) / n_traj
    return C

def calculate_cavity_correlation(cav_trajectories, j_val):
    """
    Computes the intensive cavity auto-correlation matrix.
    Expects a 2D complex array `cav_trajectories` of shape (n_trajectories, n_steps).
    """
    n_traj = cav_trajectories.shape[0]
    mean_alpha = jnp.mean(cav_trajectories, axis=0)
    fluctuations = cav_trajectories - mean_alpha
    
    # Real part of the symmetric complex product divided by j_val
    C = jnp.real(fluctuations.T @ jnp.conj(fluctuations)) / n_traj
    return C / j_val

@jax.jit
def extract_stationary_correlation(C_matrix):
    n = C_matrix.shape[0]
    rows = jnp.arange(n)
    
    def roll_row(i, row):
        return jnp.roll(row, -i)
    aligned_matrix = jax.vmap(roll_row)(rows, C_matrix)
    
    lags = jnp.arange(n)
    counts = n - lags
    c_tau_sum = jnp.sum(aligned_matrix * (rows[:, None] + lags[None, :] < n), axis=0)
    
    return c_tau_sum / jnp.where(counts > 0, counts, 1.0)

@jax.jit
def fourier_transform_correlation(c_tau, dt, w_grid):
    tau_grid = jnp.arange(len(c_tau)) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    weights = jnp.ones_like(c_tau).at[0].set(0.5).at[-1].set(0.5)
    
    S_w = 2.0 * jnp.dot(jnp.cos(w_tau), c_tau * weights) * dt
    return S_w

def measure_linear_response_fdt(keys, t_grid, p, t_pulse, target="spin", epsilon=0.01, w_max=20.0, N_w=5000):
    """
    Measures the true auto-susceptibility of either the spin or the cavity system.
    target="spin"   -> Kicks spins with B_x, measures spin response.
    target="cavity" -> Kicks cavity with E-field, measures cavity quadrature response.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    pulse_idx = jnp.searchsorted(t_grid, t_pulse)
    j_val = p['n_spins'] / 2.0

    # Base arrays (No perturbations)
    B_base = jnp.zeros((num_steps, 3)).at[:, 2].set(p['B_z'])
    E_base = jnp.zeros(num_steps)

    if target == "spin":
        B_pert = B_base.at[pulse_idx, 0].add(epsilon / dt)
        E_pert = E_base
    elif target == "cavity":
        B_pert = B_base
        E_pert = E_base.at[pulse_idx].add(epsilon / dt)

    print(f">>> Propagating Base Ensemble...")
    res_base_S, res_base_alpha = run_coupled_non_markovian_twa_bundle(
        keys=keys, t_grid=t_grid, omega_0=p['omega_0'], alpha=p['alpha'],
        omega_c=p['omega_c'], s=p['s'], T=p['T'], B_field=B_base,
        g=p['g'], n_photons_initial=p['n_photons_initial'],
        initial_direction=p['initial_direction'], batch_size=p['batch_size'],
        n_spins=p['n_spins'], w_max=w_max, N_w=N_w, cavity_drive=E_base
    )
    
    print(f">>> Propagating Perturbed Ensemble (Targeting {target})...")
    res_pert_S, res_pert_alpha = run_coupled_non_markovian_twa_bundle(
        keys=keys, t_grid=t_grid, omega_0=p['omega_0'], alpha=p['alpha'],
        omega_c=p['omega_c'], s=p['s'], T=p['T'], B_field=B_pert,
        g=p['g'], n_photons_initial=p['n_photons_initial'],
        initial_direction=p['initial_direction'], batch_size=p['batch_size'],
        n_spins=p['n_spins'], w_max=w_max, N_w=N_w, cavity_drive=E_pert
    )

    if target == "spin":
        mean_base = jnp.mean(res_base_S[:, :, 0], axis=0) / j_val
        mean_pert = jnp.mean(res_pert_S[:, :, 0], axis=0) / j_val
    elif target == "cavity":
        # Divide by sqrt(j_val) to match intensive fluctuation scales perfectly
        mean_base = jnp.mean(jnp.real(res_base_alpha), axis=0) / jnp.sqrt(j_val)
        mean_pert = jnp.mean(jnp.real(res_pert_alpha), axis=0) / jnp.sqrt(j_val)

    response = (mean_pert - mean_base) / epsilon
    return response[pulse_idx:]

@jax.jit
def fourier_transform_response(chi_tau, dt, w_grid):
    tau_grid = jnp.arange(len(chi_tau)) * dt
    w_tau = w_grid[:, None] * tau_grid[None, :]
    weights = jnp.ones_like(chi_tau).at[0].set(0.5).at[-1].set(0.5)
    
    imag_chi_w = jnp.dot(jnp.sin(w_tau), chi_tau * weights) * dt
    return imag_chi_w