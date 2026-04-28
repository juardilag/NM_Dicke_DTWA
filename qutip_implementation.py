import numpy as np
from qutip import (jmat, destroy, tensor, qeye, basis, brmesolve, 
                   spin_coherent, coherent)

def simulate_open_dicke(N=4, N_cav=10, omega_0=1.0, 
                        B_x=0.0, B_y=0.0, B_z=0.8, g=0.5, 
                        T=0.5, alpha=0.1, omega_c=5.0, s=1.0, 
                        v_spin=[0.0, 0.0, -1.0], n_photons=0.0, 
                        t_max=20.0, n_steps=500):
    """
    Simulates the open Dicke model using the Bloch-Redfield master equation 
    with a continuous, non-Markovian bath, allowing arbitrary initial states.

    Returns:
        result: QuTiP Result object.
                Expectation values map to:
                result.expect[0] -> <J_x>
                result.expect[1] -> <J_y>
                result.expect[2] -> <J_z>
                result.expect[3] -> <a^\dagger a>
    """
    j = N / 2.0

    # ==========================================
    # 1. Operators and Hamiltonian
    # ==========================================
    a = tensor(qeye(int(2*j + 1)), destroy(N_cav))
    ncav = a.dag() * a
    xc = a + a.dag()

    Jx = tensor(jmat(j, 'x'), qeye(N_cav))
    Jy = tensor(jmat(j, 'y'), qeye(N_cav))
    Jz = tensor(jmat(j, 'z'), qeye(N_cav))

    # Bare Hamiltonian with arbitrary magnetic field (H = -B \cdot J)
    H_bare = -B_x * Jx - B_y * Jy - B_z * Jz + omega_0 * ncav
    
    # Dicke Interaction: (2g/sqrt(N)) * Jx * (a + a^\dagger)
    H_int = (2.0 * g / np.sqrt(N)) * Jx * xc
    H = H_bare + H_int

    # ==========================================
    # 2. The Non-Markovian Bath Spectrum
    # ==========================================
    def J_omega(w):
        return alpha * omega_c * (np.abs(w) / omega_c)**s * np.exp(-np.abs(w) / omega_c)

    def n_th(w):
        return 1.0 / (np.exp(w / T + 1e-12) - 1.0)

    def S_spectrum(w):
        if w == 0.0:
            if s > 1: return 0.0
            if s == 1: return 2 * np.pi * alpha * omega_c * T
            else: return 0.0 
        elif w > 0.0:
            return 2 * np.pi * J_omega(w) * (n_th(w) + 1.0)
        else:
            return 2 * np.pi * J_omega(w) * n_th(np.abs(w))

    a_ops = [[xc, S_spectrum]]

    # ==========================================
    # 3. Initial State Construction
    # ==========================================
    # --- Spin State (Coherent state on the Bloch sphere) ---
    v = np.array(v_spin, dtype=float)
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-12:
        n_vec = np.array([0.0, 0.0, 1.0])  # Default to +Z if null vector passed
    else:
        n_vec = v / norm_v
        
    theta = np.arccos(n_vec[2])
    phi = np.arctan2(n_vec[1], n_vec[0])
    
    psi_spin = spin_coherent(j, theta, phi)

    # --- Cavity State ---
    if n_photons == 0.0:
        psi_cav = basis(N_cav, 0)
    else:
        # Generate coherent state |alpha> where alpha = sqrt(n)
        psi_cav = coherent(N_cav, np.sqrt(n_photons))
        
    # Joint pure initial state
    psi0 = tensor(psi_spin, psi_cav)

    # ==========================================
    # 4. Time Evolution
    # ==========================================
    tlist = np.linspace(0, t_max, n_steps)

    print(f"Solving Open Dicke Model (N={N}, s={s}, g={g})...")
    # Track all spin components and photon number
    result = brmesolve(H, psi0, tlist, a_ops, e_ops=[Jx, Jy, Jz, ncav])
    
    return tlist, result