import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import implementation functions
from dtwa_implementation import (
    run_integrated_twa_bundle,
    calculate_correlation
)

params = {
    "n_spins": 1000,
    "omega_0": 1.0,
    "B_z": 1.0,
    "g": 1.0,
    "T": 0.75,
    "alpha": 0.01,
    "omega_c": 2.5,
    "s": 0.5,
    "initial_direction": [0.01, 0.0, 0.99],
    "n_photons_initial": 0.0,
    "t_max": 150,
    "n_steps": 5_000,
    "n_trajectories": 20_000,
    "batch_size": 10_000 # Adjusted for typical memory constraints
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
# 3. Calculate Robust Observables
# ==========================================
print("Calculating Observables...")

# Normalized Jx trajectories for all realizations
sx_norm = spin_ensemble[:, :, 0] / j_val

# 1. Root Mean Square (RMS): sqrt(<Jx^2>)
rms_jx = jnp.sqrt(jnp.mean(sx_norm**2, axis=0))

# 2. Absolute Mean: <|Jx|>
abs_jx = jnp.mean(jnp.abs(sx_norm), axis=0)

# 3. Longitudinal Magnetization: <Jz> (already invariant)
mean_jz = jnp.mean(spin_ensemble[:, :, 2], axis=0) / j_val

# 4. Correlation Matrix C(t, t') using Jx
correlation = calculate_correlation(sx_norm)

# ==========================================
# 4. Plotting
# ==========================================
plt.style.use('seaborn-v0_8-white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Dynamics of Order Parameters
ax1.plot(t_grid, mean_jz, label=r'$\langle J_z \rangle / j$', color='teal', lw=2)
ax1.plot(t_grid, rms_jx, label=r'$\sqrt{\langle J_x^2 \rangle} / j$', color='crimson', linestyle='--', lw=1.5)
ax1.plot(t_grid, abs_jx, label=r'$\langle |J_x| \rangle / j$', color='darkorange', linestyle=':', lw=1.5)

ax1.set_title("DTWA Dynamics (Parity-Invariant Observables)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.set_ylim(-1.1, 1.1)
ax1.legend(loc='best')
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