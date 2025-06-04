# utility
import numpy as np
import numpy.random as npr
import qutip as qt
import math
import torch

def calc_bitdist(a, b, N):
    """
    TODO
    """
    diffs = a ^ b
    return sum((diffs >> i) & 1 for i in range(N))

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
    return input

def generate_input_samples(N, samples):
    """
    Given some number of samples to generate input for, creates (|samples|, N) size torch
    tensor with the ith row corresponding to the binary representation of the ith sample taken

    Args:
        N (int): number of bits to represent integers with
        samples (list(int)): integer representations of the basis states to generate input for
    """
    return torch.tensor([generate_state_array(x, N) for x in samples]).to(torch.float32)

def generate_input_samples_torch(samples, N):
    """
    TODO
    """
    powers = torch.arange(N, dtype=torch.long)               # shape: (N,)
    bits = (samples.unsqueeze(1) >> powers) & 1           # shape: (B, N)
    return bits.to(torch.float32)

def log_amp_phase(nn_output):
    """
    Given (N, 2) shape output from neural network with the first column representing log(amp)
    and the second column representing phase, returns N dimensional tensor with each element
    representing the complex amplitude of the corresponding row

    Args:
        nn_output (torch.tensor): output from torch model with each row representing log(amp), phase output
    """
    return torch.exp(nn_output[:, 0] + 1.j * nn_output[:, 1])

def bitflip_x(x, N, flips):
    """
    Returns new integer with flips random bitflips made

    Args:
        x (int): state to change
        N (int): number of bits to represent ints with
        flips (int): number of random flips to be made
    """
    new_x = x
    for _ in range(flips):
        new_x = x ^ (1 << npr.randint(0, N))
    return new_x

def bitflip_batch(xs, N, flips):
    """
    Vectorized random bit flips on a batch of integers.

    Args:
        xs (Tensor): shape (B,), integers
        N (int): number of bits
        flips (int): number of random bit flips per element

    Returns:
        Tensor of shape (B,), integers after bit flips
    """
    B = xs.shape[0]
    xs = xs.clone()

    # Generate random bit indices for each flip and sample
    bit_indices = torch.randint(0, N, size=(B, flips))

    # Compute bitmasks: 1 << bit index
    bitmasks = (1 << bit_indices)  # shape: (B, flips)

    flip_masks = bitmasks[:, 0]
    for i in range(1, flips):
        flip_masks = flip_masks ^ bitmasks[:, i]

    return xs ^ flip_masks