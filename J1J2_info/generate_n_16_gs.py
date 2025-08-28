import numpy as np
import qutip as qt
import pandas as pd
import qutip as qt
import sys
import os

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


my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

N=16
J1=1
gses = []

os.makedirs(f"N{N}", exist_ok=True)

J2s=np.linspace(0, 0.5-0.005, 100)
my_j2s = J2s[my_task_id:len(J2s):num_tasks]

for J2 in my_j2s:
    h = J1J2_hamiltonian(N, J1, J2)
    eigs = h.eigenstates()
    gs = eigs[1][0]
    gse = eigs[0][0]
    gses.append(gse)
    qt.qsave(gs, f'N{N}/n{N}gs_J2_{round(J2, 3)}')
df = pd.DataFrame({'J2':my_j2s, 'GSE':gses})
df.to_csv(f'N{N}/gs_energies_{my_task_id}.csv')