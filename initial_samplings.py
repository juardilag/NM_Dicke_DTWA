import jax.numpy as jnp
import jax

def discrete_spin_sampling_factorized(key, initial_direction, n_spins=1):
    """
    Tu lógica original generalizada para un modelo colectivo de N espines.
    S(0) = N * Mean + F1 * perp1 + F2 * perp2
    Las fluctuaciones transversales F1 y F2 son la suma de N variables +/- 1.
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Dirección media
    mean_vec = initial_direction / (jnp.linalg.norm(initial_direction) + 1e-12)
    
    # 2. Gram-Schmidt para encontrar ejes perpendiculares
    # Usamos jnp.where en lugar de if para que sea compatible con JIT
    v = jnp.where(jnp.abs(mean_vec[0]) < 0.9, 
                  jnp.array([1.0, 0.0, 0.0]), 
                  jnp.array([0.0, 1.0, 0.0]))
    
    perp1 = jnp.cross(mean_vec, v)
    perp1 = perp1 / (jnp.linalg.norm(perp1) + 1e-12)
    perp2 = jnp.cross(mean_vec, perp1)
    
    # 3. Fluctuaciones discretas +/- 1 para N espines
    # Generamos N variables aleatorias para cada eje transversal y las sumamos
    flips1 = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=(n_spins,)) - 1.0
    flips2 = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=(n_spins,)) - 1.0
    
    f1 = jnp.sum(flips1)
    f2 = jnp.sum(flips2)
    
    # 4. Construcción del espín macroscópico
    # El campo medio escala linealmente con N, las fluctuaciones escalan como sqrt(N)
    s_init = (n_spins * mean_vec) + (f1 * perp1) + (f2 * perp2)
    
    return s_init