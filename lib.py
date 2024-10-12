import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import math
import torch
import torch.nn as nn

def TFIM_hamiltonian(N, J, gamma):
    # uses ring shape
    id = qt.qeye(2)
    z = qt.sigmaz()
    x = qt.sigmax()
    sxi = []
    szi = []
    for i in range(N):
        sxi.append(qt.tensor([id] * i + [x] + [id] * (N - i - 1)))
        szi.append(qt.tensor([id] * i + [z] + [id] * (N - i - 1)))
    return -J * sum(szi[i] * szi[i + 1] for i in range(N - 1)) - J * szi[N - 1] * szi[0] - gamma * sum(sxi[i] for i in range(N))

# magnetization

def count_magnetization(state): 
    """
    With integer encoding of state such that 1 in binary representation 
    is spin down and 0 is spin up. Returns number of spin downs.
    """
    if state == 0: return 0
    return sum((state >> n) & 1 for n in range(0, int(np.log2(state)) + 1))

def absolute_z_magnetization(N, psi):
    total_mag = 0
    dim = 2 ** N
    for basis_vec in range(dim):
        down_spins = count_magnetization(basis_vec)
        total_mag += abs(psi[basis_vec]) ** 2 * abs(down_spins - (N - down_spins))
    return total_mag

def signed_x_magnetization(N, state):
    total_mag = 0
    return total_mag

# manual NN training

def TFIM_multiply(psi, phi, N, J, Gamma):
    dim = 2 ** N
    for state in range(dim):
        jtotal = 0
        for site in range(N - 1):
            jtotal += J if ((state >> site) ^ (state >> (site + 1))) & 1 else -J 
        jtotal += J if ((state >> (N - 1)) ^ (state >> 0)) & 1 else -J
        phi[state] = jtotal * psi[state]
    
    for state in range(dim):
        for site in range(N):
            flipped_state = state ^ (1 << site)
            phi[flipped_state] -= Gamma*psi[state]

def TFIM_expectation_from_array(psi, N, J, Gamma):
    dim = 2 ** N
    phi = np.zeros(dim, dtype=np.complex128)
    psi = np.array(psi)
    bra_psi = psi.reshape((1, -1)).conj()
    TFIM_multiply(psi, phi, N, J, Gamma)
    return (bra_psi @ phi.reshape((-1, 1)))[0][0]

def generate_state_vector(state_num, N):
    x = np.zeros((N, 1))
    for n in range(N):
        x[n][0] = (state_num >> n) & 1
    return x

def sigmoid(x):
    return np.array([math.copysign(1, n[0]) * 1 if abs(n[0]) > 10 else 1/(1 + np.exp(-n[0])) for n in x]).reshape((-1, 1))

def compute_NN_manual(weights, layers, N, J, Gamma):
    # weights as 1D array with arrays and vectors flattened
    # those uses Re, Im output
    results = []
    dim = 2 ** N

    weights = np.array(weights)
    weight_arr = []
    bias_arr = []
    total = 0
    for i in range(1, len(layers)):
        weight_arr.append(weights[total:(total + layers[i] * layers[i - 1])].reshape((layers[i], layers[i - 1])))
        total += layers[i] * layers[i - 1]
        bias_arr.append(weights[total:(total + layers[i])].reshape((-1, 1)))
        total += layers[i]
    for state in range(dim):
        vector = generate_state_vector(state, N)
        for i in range(len(weight_arr)):
            vector = sigmoid(weight_arr[i] @ vector + bias_arr[i])
        results.append(vector[0][0] + vector[1][0] * 1.j)
    
    mag = sum(abs(n) ** 2 for n in results)

    expect = TFIM_expectation_from_array(results, N, J, Gamma) / mag
    return expect.real 

def number_weights(layers):
    total = 0
    for i in range(1, len(layers)):
        total += layers[i] * layers[i - 1] + layers[i]
    return total

def weights_to_ground_state(res, N, layers):
    #Re, Im output
    weights = res.x

    results = []
    dim = 2 ** N

    weights = np.array(weights)
    weight_arr = []
    bias_arr = []
    total = 0
    for i in range(1, len(layers)):
        weight_arr.append(weights[total:(total + layers[i] * layers[i - 1])].reshape((layers[i], layers[i - 1])))
        total += layers[i] * layers[i - 1]
        bias_arr.append(weights[total:(total + layers[i])].reshape((-1, 1)))
        total += layers[i]
    for state in range(dim):
        vector = generate_state_vector(state, N)
        for i in range(len(weight_arr)):
            vector = sigmoid(weight_arr[i] @ vector + bias_arr[i])
        results.append(vector[0][0] + vector[1][0] * 1.j)

    mag = sum(abs(n) ** 2 for n in results)
    gs = np.array(results) / np.sqrt(mag)
    return gs

# torch stuff

def TFIM_expectation_from_torch(nn_output, vars, output_to_psi):
    N, J, Gamma = vars
    dim = 2 ** N
    psi = output_to_psi(nn_output)
    mag = sum(abs(n) ** 2 for n in psi)
    phi = torch.zeros(dim, dtype = torch.complex64)
    bra_psi = psi.reshape((1, -1)).conj()
    TFIM_multiply(psi, phi, N, J, Gamma)

    res = (bra_psi @ phi.reshape((-1, 1)))[0][0].real

    return res / mag

def generate_state_1D(state_num, N):
    x = np.zeros(N)
    for n in range(N):
        x[n] = (state_num >> n) & 1
    return x

def generate_input_torch(N):
    dim = 2 ** N
    input = np.array([generate_state_1D(state, N) for state in range(dim)])
    input = torch.tensor(input, dtype = torch.float32)
    return input

def model_to_ground_state(model, input, output_to_psi):
    """
    output to psi takes a (2^N, 2) output from a torch network and returns
    a 2^N size statevector
    """
    pred = model(input)
    pred_gs = output_to_psi(pred)
    mag = sum(abs(n) ** 2 for n in pred_gs)
    pred_gs = qt.Qobj(pred_gs.data / math.sqrt(mag))
    return pred_gs
