import qutip as qt
import numpy as np
import math
import torch

# utility

def calc_bitdist(a, b, N):
    diffs = a ^ b
    return sum((diffs >> i) & 1 for i in range(N))

def generate_adjacencies(state, N):
    adjacents = []
    for i in range(0, N):
        adjacents.append(state ^ (1 << i))
    return adjacents

# transverse field Ising model

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

def z_to_x(N, psi):
    """
    Converts Qobj from z basis to x basis

    Args:
        N (int): number of qubits
        psi (Qobj): quantum state
    """
    xplus = 1/math.sqrt(2) * (qt.basis(2, 0) + qt.basis(2, 1))
    xminus = 1/math.sqrt(2) * (qt.basis(2, 0) - qt.basis(2, 1))
    dim = 2 ** N
    basis = []
    for i in range(dim):
        basis.append(qt.tensor([[xplus, xminus][(i >> (N - k - 1)) & 1] for k in range(0, N)]))
    return psi.transform(basis)

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
    
# magnetization

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

# manual NN training

def TFIM_multiply(psi, phi, N, J, Gamma):
    """
    Outputs H|psi> into vector phi

    Args:
        psi: 1D array representing statevector
        phi: 1D array to output results to
        N, J, Gamma: parameters in Hamiltonian
    """
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
    """
    Calculates <psi|H|psi>/<psi|psi>
    """
    dim = 2 ** N
    phi = np.zeros(dim, dtype=np.complex128)
    psi = np.array(psi)
    bra_psi = psi.reshape((1, -1)).conj()
    TFIM_multiply(psi, phi, N, J, Gamma)
    return (bra_psi @ phi.reshape((-1, 1)))[0][0]

def generate_state_vector(state_num, N):
    """
    Returns binary representation of integer state as column vector
    """
    x = np.zeros((N, 1))
    for n in range(N):
        x[n][0] = (state_num >> n) & 1
    return x

def sigmoid(x):
    """
    Returns cut-off sigmoid of np.array x
    """
    return np.array([math.copysign(1, n[0]) * 1 if abs(n[0]) > 10 else 1/(1 + np.exp(-n[0])) for n in x]).reshape((-1, 1))

def compute_NN_manual(weights, layers, N, J, Gamma):
    """
    Computes TFIM expectation from neural network. 
    Assumes neural network uses sigmoid at every layer and 
    outputs two nodes Re[psi[i]] and Im[psi[i]]

    Args:
        weights: flattened 1D array of all weights and biases ordered
        layers: structure of neural network
        N, J, Gamma: Hamiltonian parameters
    
    Returns:
        Expectation value from forward pass
    """
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
    
    mag2 = sum(abs(n) ** 2 for n in results)
    expect = TFIM_expectation_from_array(results, N, J, Gamma) / mag2
    return expect.real 

def number_weights(layers):
    """
    Calculates number of weights + biases necessary given layer counts
    of neural network
    """
    total = 0
    for i in range(1, len(layers)):
        total += layers[i] * layers[i - 1] + layers[i]
    return total

def weights_to_ground_state(res, N, layers):
    """
    Takes results from optimized neural network weights
    and runs forward pass to generate final ground state

    Args:
        res: results from scipy.optimize.minimize (preferred method = 'Powell')
        N: number of qubits
        layers: layer counts for neural network

    Returns:
        (2^N, 1) np.array as final ground state vector
    """
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

    mag = sum(abs(n) ** 2 for n in results) ** 0.5
    gs = np.array(results) / mag
    return gs

# torch stuff

def TFIM_expectation_from_torch(nn_output, vars, output_to_psi):
    """
    Args:
        nn_output: torch.tensor of shape (2^N, N) output from torch model
        vars: tuple of (N, J, Gamma)
        output_to_psi: function converting nn_output to 2^N size 
        statevector
    """
    N, J, Gamma = vars
    dim = 2 ** N
    psi = output_to_psi(nn_output)
    mag2 = torch.norm(psi, 2) ** 2
    # mag = sum(abs(n) ** 2 for n in psi) # make these torch functions, use torch.norm
    phi = torch.zeros(dim, dtype = torch.complex64)
    bra_psi = psi.reshape((1, -1)).conj()
    TFIM_multiply(psi, phi, N, J, Gamma)

    res = (bra_psi @ phi.reshape((-1, 1)))[0][0].real

    return res / mag2

def generate_state_1D(state_num, N):
    """
    Returns binary representation of integer state as 1D vector
    """
    x = np.zeros(N)
    for n in range(N):
        x[n] = (state_num >> n) & 1
    return x

def generate_state_array(state_num, N):
    """
    Returns binary representation of integer state as 1D list
    """
    x = []
    for n in range(N):
        x.append((state_num >> n) & 1)
    return x

def generate_input_torch(N):
    """
    Generates (2^N, N) shape tensor to feed into torch network
    """
    dim = 2 ** N
    input = np.array([generate_state_1D(state, N) for state in range(dim)])
    input = torch.tensor(input, dtype = torch.float32)
    # input = torch.tensor([generate_state_array(state, N) for state in range(dim)])
    return input

def model_to_ground_state(model, input, output_to_psi):
    """
    Args:
        model: trained torch model
        input: tensor of all binary states of size (2^N, N) in order
        output_to_psi: function that takes a (2^N, 2) tensor output 
        from a torch network and returns a 2^N size tensor

    Returns:
        Returns normalized Qobj state derived from model
    """
    pred = model(input)
    pred_gs = output_to_psi(pred)
    mag = math.sqrt(sum(abs(n) ** 2 for n in pred_gs))
    pred_gs = qt.Qobj(pred_gs.data / mag)
    return pred_gs

def train_model_to_gs(model, generate_y_pred, loss_fn, num_epochs, data_rate = 50):
    """
    Trains model to find the ground state

    Args:
        model: torch model to be trained
        generate_y_pred (func): that takes in model and outputs psi
        loss_fn (func): takes in psi and outputs loss
        num_epochs (int): steps to train for
        data_rate (int): collects loss data every data_rate epochs
    
    Returns:
        tuple of epochs at which loss was collected and loss at those epochs
    """
    epochs = []
    loss_data = []
    optimizer = torch.optim.SGD(model.parameters(), lr = 2)
    for epoch in range(num_epochs):
        y_pred = generate_y_pred(model)
        loss = loss_fn(y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % data_rate == 0:
            loss_data.append(loss.item())
            epochs.append(epoch)
    return epochs, loss_data

# MC methods

def TFIM_expectation_using_locals(sampled_vector, N, J, Gamma, model, output_to_psi):
    psi_calcs = {}
    def psi(x):
        if x in sampled_vector.values:
            return sampled_vector.values[x]
        if x in psi_calcs:
            return psi_calcs[x]
        tens = torch.tensor([generate_state_array(x, N)], dtype = torch.float32)
        complex_amp = output_to_psi(model(tens))[0]
        psi_calcs[x] = complex_amp
        return complex_amp
    total_num = 0
    total_denom = 0
    for basis_state in sampled_vector.values:
        eloc = 0
        for adjacency in generate_adjacencies(basis_state, N):
            eloc += calc_H_elem(N, J, Gamma, basis_state, adjacency) * psi(adjacency) / psi(basis_state)
        eloc += calc_H_elem(N, J, Gamma, basis_state, basis_state)
        amp = abs(psi(basis_state)) ** 2
        total_num += amp * eloc
        total_denom += amp
    return (total_num / total_denom).real


    