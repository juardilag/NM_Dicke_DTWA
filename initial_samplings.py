import jax.numpy as jnp
import jax

def discrete_spin_sampling_factorized(key, initial_direction, n_spins=1):
    k1, k2 = jax.random.split(key)
    
    # 1. Directional normalization
    initial_direction = jnp.array(initial_direction, dtype=float)
    mean_vec = initial_direction / (jnp.linalg.norm(initial_direction) + 1e-12)
    
    # 2. Perpendicular axes
    v = jnp.where(jnp.abs(mean_vec[0]) < 0.9, 
                  jnp.array([1.0, 0.0, 0.0]), 
                  jnp.array([0.0, 1.0, 0.0]))
    
    perp1 = jnp.cross(mean_vec, v)
    perp1 = perp1 / (jnp.linalg.norm(perp1) + 1e-12)
    perp2 = jnp.cross(mean_vec, perp1)
    
    # 3. Discrete fluctuations (+/- 1)
    flips1 = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=(n_spins,)) - 1.0
    flips2 = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=(n_spins,)) - 1.0
    
    f1 = jnp.sum(flips1)
    f2 = jnp.sum(flips2)
    
    # Returns vector of length N (Pauli scaling)
    s_init = (n_spins * mean_vec) + (f1 * perp1) + (f2 * perp2)
    
    return s_init