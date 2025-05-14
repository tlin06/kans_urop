# transverse field Ising model
import numpy as np
import qutip as qt
from utils import calc_bitdist

def TFIM_hamiltonian(N, J, gamma):
    """
    Generates sparse Qobj for TFIM system with ring shape
    """
    id = qt.qeye(2)
    z = qt.sigmaz()
    x = qt.sigmax()
    sxi = []
    szi = []
    for i in range(N):
        sxi.append(qt.tensor([id] * i + [x] + [id] * (N - i - 1)))
        szi.append(qt.tensor([id] * i + [z] + [id] * (N - i - 1)))
    return -J * sum(szi[i] * szi[i + 1] for i in range(N - 1)) - J * szi[N - 1] * szi[0] - gamma * sum(sxi[i] for i in range(N))

def ground_state_energy(h, N):
    return ground_state_energy_per_site(h, N) * N

def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)

def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))

def generate_adjacencies(state, N):
    adjacents = []
    for i in range(0, N):
        adjacents.append(state ^ (1 << i))
    adjacents.append(state)
    return adjacents

def calc_H_elem(N, J, Gamma, i, j):
    """
    Calculates <i|H|j> 
    """
    if i != j:
        if calc_bitdist(i, j, N) == 1:
            return -Gamma
        return 0
    total = 0
    for index in range(0, N - 1):
        total += ((i >> index) ^ (i >> (index + 1))) & 1
    total += ((i >> (N - 1)) ^ (i >> 0)) & 1
    return - ((N - total) * J + total * -J)