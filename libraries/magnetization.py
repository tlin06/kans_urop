# magnetization
import numpy as np
from tfim_functions import z_to_x

def count_magnetization(state): 
    """
    Counts number of 1s in binary representation of some integer
    With integer encoding of state such that 1 in binary representation 
    is spin down and 0 is spin up. Returns number of spin downs.
    """
    if state == 0: return 0
    return sum((state >> n) & 1 for n in range(0, int(np.log2(state)) + 1))

def z_magnetization(N, psi, signed = True):
    """
    Calculates the amount of spin in a z-basis statevector
    Signed is up spins minus down spins
    Unsigned is absolute value of up spins minus down spins for every state
    
    We could've also replaced this with expectation value of 
    sigmaz_i summed over all indexes for signed, and 
    |0><0|_i summed over all indexes for unsigned

    """
    total_mag = 0
    dim = 2 ** N
    for basis_vec in range(dim):
        down_spins = count_magnetization(basis_vec)
        mag = abs(psi[basis_vec]) ** 2 * ((N - down_spins) - down_spins)
        if not signed: mag = abs(mag)
        total_mag += mag
    return total_mag

def x_magnetization(N, psi, signed = True):
    """
    Calculates the amount of spin in the x direction in a z-basis
    statevector. 
    """
    x_psi = z_to_x(N, psi)
    return z_magnetization(N, x_psi, signed = signed)