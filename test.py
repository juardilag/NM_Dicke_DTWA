import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import only necessary TWA functions
from dtwa_implementation import (
    run_integrated_twa_bundle,
    calculate_correlation
)

params = {
    "n_spins": 1000,
    "omega_0": 1.0,
    "B_x": 0.0,
    "B_y": 0.0,
    "B_z": 1.0,
    "g": 1.0,
    "T": 0.75,
    "alpha": 0.01,
    "omega_c": 2.5,
    "s": 1.0,
    "initial_direction": [0.01, 0.0, 0.99],
    "n_photons_initial": 0.0,
    "t_max": 80,
    "n_steps": 5_000,
    "n_trajectories": 10_000,
    "batch_size": 25_000
}

# ==========================================
# 1. Setup
# ==========================================
t_grid = jnp.linspace(0, params["t_max"], params["n_steps"])
B_field = jnp.zeros((params["n_steps"], 3)).at[:, 2].set(params["B_z"])

initial_dir_array = jnp.array(params["initial_direction"])
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, params["n_trajectories"])
j_val = params["n_spins"] / 2.0

# ==========================================
# 2. Run Simulation
# ==========================================
print("Running TWA Simulation...")
spin_ensemble = run_integrated_twa_bundle(
    keys=keys, t_grid=t_grid, omega_0=params["omega_0"], alpha=params["alpha"],
    omega_c=params["omega_c"], s=params["s"], T=params["T"], B_field=B_field,
    g=params["g"], n_photons_initial=params["n_photons_initial"],
    initial_direction=initial_dir_array, batch_size=params["batch_size"],
    n_spins=params["n_spins"], use_noise=True, use_sampling=True
)

# ==========================================
# 3. Calculate Observables
# ==========================================
print("Calculating Dynamics and Correlations...")

# Mean trajectories (Dynamics)
mean_traj = jnp.mean(spin_ensemble, axis=0) / j_val

# Symmetric Correlation: C(t, t')
sx_all_times = spin_ensemble[:, :, 2] / j_val
correlation = calculate_correlation(sx_all_times)

# ==========================================
# 4. Plotting
# ==========================================
plt.style.use('seaborn-v0_8-white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Dynamics
ax1.plot(t_grid, mean_traj[:, 0], label=r'$\langle J_x \rangle / j$', lw=1.5)
ax1.plot(t_grid, mean_traj[:, 1], label=r'$\langle J_y \rangle / j$', lw=1.5)
ax1.plot(t_grid, mean_traj[:, 2], label=r'$\langle J_z \rangle / j$', lw=1.5)
#ax1.set_title("Spin Dynamics (Expectation Values)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Normalization")
ax1.set_ylim(-1.1, 1.1)
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# Panel 2: Correlation Matrix
cf = ax2.imshow(correlation, origin='lower', cmap='plasma', 
                extent=[0, params["t_max"], 0, params["t_max"]], aspect='auto')
ax2.set_title(r"Symmetric Correlation $C(t, t')$")
ax2.set_xlabel(r"$t'$")
ax2.set_ylabel(r"$t$")
fig.colorbar(cf, ax=ax2)

plt.tight_layout()
plt.show()