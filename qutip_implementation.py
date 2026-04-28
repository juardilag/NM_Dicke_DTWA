import numpy as np
from qutip import jmat, destroy, tensor, qeye, basis, brmesolve

def simulate_open_dicke(N=4, N_cav=10, omega_0=1.0, B_z=0.8, B_x=0.0, g=0.5, 
                        T=0.5, alpha=0.1, omega_c=5.0, s=1.0, 
                        t_max=20.0, n_steps=500):
    """
    Simulates the open Dicke model using the Bloch-Redfield master equation 
    with a continuous, non-Markovian bath.

    Returns:
        result: QuTiP Result object containing the expectation values.
                Expectation values map to:
                result.expect[0] -> <J_z>
                result.expect[1] -> <J_x>
                result.expect[2] -> <a^\dagger a>
    """
    j = N / 2.0

    # ==========================================
    # Operators and Hamiltonian
    # ==========================================
    a = tensor(qeye(int(2*j + 1)), destroy(N_cav))
    ncav = a.dag() * a
    xc = a + a.dag()

    Jx = tensor(jmat(j, 'x'), qeye(N_cav))
    Jy = tensor(jmat(j, 'y'), qeye(N_cav))
    Jz = tensor(jmat(j, 'z'), qeye(N_cav))

    H_bare = -B_z * Jz - B_x * Jx + omega_0 * ncav
    H_int = (2.0 * g / np.sqrt(N)) * Jx * xc
    H = H_bare + H_int

    # ==========================================
    # The Non-Markovian Bath Spectrum
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
    # Initial State and Time Evolution
    # ==========================================
    # Start with spins all pointing down (-Z) and vacuum in the cavity
    psi0_spin = basis(int(2*j + 1), int(2*j)) 
    psi0_cav = basis(N_cav, 0)                
    psi0 = tensor(psi0_spin, psi0_cav)

    tlist = np.linspace(0, t_max, n_steps)

    # Solve using the Bloch-Redfield tensor formalism
    print(f"Solving Open Dicke Model (N={N}, s={s}, g={g})...")
    result = brmesolve(H, psi0, tlist, a_ops, e_ops=[Jz, Jx, ncav])
    
    return result