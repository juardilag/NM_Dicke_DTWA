import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from dtwa_implementation import run_integrated_twa_bundle

# ==========================================
# 1. Configuration
# ==========================================
params = {
    "n_spins": 500,
    "omega_0": 1.0,
    "B_z": 1.0,
    "T": 0.75,
    "alpha": 0.1,
    "omega_c": 2.5,
    "s": 1.0,
    "initial_direction": [0.001, 0.0, 0.999],
    "n_photons_initial": 0.0,
    "t_max": 200,
    "n_steps": 2500, # Balanced for sweep performance
    "n_trajectories": 2000, 
    "batch_size": 2000
}

t_grid = jnp.linspace(0, params["t_max"], params["n_steps"])
B_field = jnp.zeros((params["n_steps"], 3)).at[:, 2].set(params["B_z"])
initial_dir_array = jnp.array(params["initial_direction"])
j_val = params["n_spins"] / 2.0

key = jax.random.PRNGKey(42)
dtwa_keys = jax.random.split(key, params["n_trajectories"])
mean_key = jax.random.split(key, 1)

g_values = jnp.linspace(0.2, 1.5, 40) 

# Storage for multiple order parameters
mf_x, mf_z = [], []
dtwa_abs_x, dtwa_rms_x, dtwa_z = [], [], []

# ==========================================
# 2. Dual Coupling Sweep
# ==========================================
print(f"Starting Dual Sweep over {len(g_values)} points...")

for g_curr in tqdm(g_values):
    # --- A. Mean-Field (Deterministic) ---
    mf_ensemble = run_integrated_twa_bundle(
        keys=mean_key, t_grid=t_grid, omega_0=params["omega_0"], alpha=params["alpha"],
        omega_c=params["omega_c"], s=params["s"], T=params["T"], B_field=B_field,
        g=g_curr, n_photons_initial=params["n_photons_initial"],
        initial_direction=initial_dir_array, batch_size=1, n_spins=params["n_spins"],
        use_noise=False, use_sampling=False
    )
    
    # --- B. DTWA (Stochastic Ensemble) ---
    dtwa_ensemble = run_integrated_twa_bundle(
        keys=dtwa_keys, t_grid=t_grid, omega_0=params["omega_0"], alpha=params["alpha"],
        omega_c=params["omega_c"], s=params["s"], T=params["T"], B_field=B_field,
        g=g_curr, n_photons_initial=params["n_photons_initial"],
        initial_direction=initial_dir_array, batch_size=params["batch_size"],
        n_spins=params["n_spins"], use_noise=True, use_sampling=True
    )
    
    last_steps = int(params["n_steps"] * 0.05)
    
    # Extract Mean-Field Steady State
    mf_x.append(jnp.abs(jnp.mean(mf_ensemble[0, -last_steps:, 0])) / j_val)
    mf_z.append(jnp.mean(mf_ensemble[0, -last_steps:, 2]) / j_val)
    
    # Extract DTWA Steady State (Handling the ensemble distribution)
    # 1. Absolute Mean over trajectories
    abs_x = jnp.mean(jnp.abs(dtwa_ensemble[:, -last_steps:, 0])) / j_val
    dtwa_abs_x.append(abs_x)
    
    # 2. RMS (Square root of the second moment)
    rms_x = jnp.sqrt(jnp.mean(dtwa_ensemble[:, -last_steps:, 0]**2)) / j_val
    dtwa_rms_x.append(rms_x)
    
    # 3. Longitudinal Jz
    z_val = jnp.mean(dtwa_ensemble[:, -last_steps:, 2]) / j_val
    dtwa_z.append(z_val)

# ==========================================
# 3. Two-Window Comparison Plot
# ==========================================
plt.style.use('seaborn-v0_8-white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Theoretical gc line
g_c = jnp.sqrt(params["omega_0"] * params["B_z"]) / 2.0

# Window 1: Transverse Order Parameters (Jx)
ax1.plot(g_values, mf_x, 'o-', color='teal', label=r'MF: $|\langle J_x \rangle|/j$')
ax1.plot(g_values, dtwa_abs_x, 's--', color='darkorange', label=r'DTWA: $\langle |J_x| \rangle/j$')
ax1.plot(g_values, dtwa_rms_x, '^:', color='crimson', label=r'DTWA: $\sqrt{\langle J_x^2 \rangle}/j$')
ax1.axvline(g_c, color='gray', linestyle='--', label=r'Closed $g_c$ limit')
ax1.set_title(r"Transverse Order Parameter (Symmetry Breaking)")
ax1.set_xlabel(r"Coupling $g$")
ax1.set_ylabel(r"Amplitude")
ax1.legend()
ax1.grid(alpha=0.2)

# Window 2: Longitudinal Magnetization (Jz)
ax2.plot(g_values, mf_z, 'o-', color='teal', label=r'MF: $\langle J_z \rangle/j$')
ax2.plot(g_values, dtwa_z, 's--', color='darkorange', label=r'DTWA: $\langle J_z \rangle/j$')
ax2.axvline(g_c, color='gray', linestyle='--', label=r'Closed $g_c$ limit')
ax2.set_title(r"Longitudinal Magnetization")
ax2.set_xlabel(r"Coupling $g$")
ax2.set_ylabel(r"Value")
ax2.legend()
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.show()