import numpy as np
import qutip as qt
import pandas as pd
import qutip as qt

def J1J2_hamiltonian(N, j1, j2):
    id = qt.qeye(2)
    x = qt.sigmax()
    y = qt.sigmay()
    z = qt.sigmaz()
    sxi = []; syi = []; szi = []
    for i in range(N):
        sxi.append(qt.tensor([id] * i + [x] + [id] * (N - i - 1)))
        syi.append(qt.tensor([id] * i + [y] + [id] * (N - i - 1)))
        szi.append(qt.tensor([id] * i + [z] + [id] * (N - i - 1)))
    sis = [sxi, syi, szi]
    J1_term = j1 * (sum(sis[coord][i] * sis[coord][i + 1] for coord in range(len(sis)) for i in range(N - 1)) + sum(sis[coord][N - 1] * sis[coord][0] for coord in range(len(sis))))
    J2_term = j2 * (sum(sis[coord][i] * sis[coord][i + 2] for coord in range(len(sis)) for i in range(N - 2)) + sum(sis[coord][N - 2] * sis[coord][0] for coord in range(len(sis))) + sum(sis[coord][N - 1] * sis[coord][1] for coord in range(len(sis))))
    return J1_term + J2_term

J2s = 1/ np.linspace(0 + 2 / 100, 2, 100)
J1 = 1

N=8
gses = []
for J2 in J2s:
    h = J1J2_hamiltonian(N, J1, J2)
    eigs = h.eigenstates()
    gs = eigs[1][0]
    gse = eigs[0][0]
    gses.append(gse)
    qt.qsave(gs, f'large_J2/N{N}/n{N}gs_J2_{round(J2, 4)}')
df = pd.DataFrame({'J2':J2s, 'GSE':gses})
df.to_csv(f'large_J2/N{N}/gs_energies.csv')

# N=10
# gses = []
# for J2 in J2s:
#     h = J1J2_hamiltonian(N, J1, J2)
#     eigs = h.eigenstates()
#     gs = eigs[1][0]
#     gse = eigs[0][0]
#     gses.append(gse)
#     qt.qsave(gs, f'large_J2/N{N}/n{N}gs_J2_{round(J2, 4)}')
# df = pd.DataFrame({'J2':J2s, 'GSE':gses})
# df.to_csv(f'large_J2/N{N}/gs_energies.csv')

# N=12
# gses = []
# for J2 in J2s:
#     h = J1J2_hamiltonian(N, J1, J2)
#     eigs = h.eigenstates()
#     gs = eigs[1][0]
#     gse = eigs[0][0]
#     gses.append(gse)
#     qt.qsave(gs, f'large_J2/N{N}/n{N}gs_J2_{round(J2, 4)}')
# df = pd.DataFrame({'J2':J2s, 'GSE':gses})
# df.to_csv(f'large_J2/N{N}/gs_energies.csv')
