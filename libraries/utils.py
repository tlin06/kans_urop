# utility
import numpy as np
import numpy.random as npr
import qutip as qt
import math
import torch
import sympy
import inspect

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

def amp_phase(nn_output):
    """
    TODO
    """
    return nn_output[:, 0] * torch.exp(1.j * 2 * np.pi * nn_output[:, 1])

def reim(nn_output):
    """
    TODO
    """
    return nn_output[:, 0] + 1.j * nn_output[:, 1]

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

def symbolic_formula(self, var=None, normalizer=None, output_normalizer = None):
    '''
    get symbolic formula. modified from KAN package to allow manually fixed activations

    Args:
    -----
        var : None or a list of sympy expression
            input variables
        normalizer : [mean, std]
        output_normalizer : [mean, std]
        
    Returns:
    --------
        None

    Example
    -------
    >>> from kan import *
    >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=3)
    >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
    >>> model.auto_symbolic()
    >>> model.symbolic_formula()[0][0]
    '''
    
    symbolic_acts = []
    symbolic_acts_premult = []
    x = []

    def ex_round(ex1, n_digit):
        ex2 = ex1
        for a in sympy.preorder_traversal(ex1):
            if isinstance(a, sympy.Float):
                ex2 = ex2.subs(a, round(a, n_digit))
        return ex2

    # define variables
    if var == None:
        for ii in range(1, self.width[0][0] + 1):
            exec(f"x{ii} = sympy.Symbol('x_{ii}')")
            exec(f"x.append(x{ii})")
    elif isinstance(var[0], sympy.Expr):
        x = var
    else:
        x = [sympy.symbols(var_) for var_ in var]

    x0 = x

    if normalizer != None:
        mean = normalizer[0]
        std = normalizer[1]
        x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

    symbolic_acts.append(x)

    for l in range(len(self.width_in) - 1):
        num_sum = self.width[l + 1][0]
        num_mult = self.width[l + 1][1]
        y = []
        for j in range(self.width_out[l + 1]):
            yj = 0.
            for i in range(self.width_in[l]):
                a, b, c, d = self.symbolic_fun[l].affine[j, i]
                sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                try:
                    yj += c * sympy_fun(a * x[i] + b) + d
                except:
                    sympy_fun = sympy.Function(f'({inspect.getsource(sympy_fun)})')
                    yj += c * sympy_fun(a * x[i] + b) + d
            yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
            y.append(yj)
                
        symbolic_acts_premult.append(y)
            
        mult = []
        for k in range(num_mult):
            if isinstance(self.mult_arity, int):
                mult_arity = self.mult_arity
            else:
                mult_arity = self.mult_arity[l+1][k]
            for i in range(mult_arity-1):
                if i == 0:
                    mult_k = y[num_sum+2*k] * y[num_sum+2*k+1]
                else:
                    mult_k = mult_k * y[num_sum+2*k+i+1]
            mult.append(mult_k)
            
        y = y[:num_sum] + mult
        
        for j in range(self.width_in[l+1]):
            y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]
        
        x = y
        symbolic_acts.append(x)

    if output_normalizer != None:
        output_layer = symbolic_acts[-1]
        means = output_normalizer[0]
        stds = output_normalizer[1]

        assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
        assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
        
        output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
        symbolic_acts[-1] = output_layer


    self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
    self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

    out_dim = len(symbolic_acts[-1])
    #return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
    return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

def get_nonzero_states(N, gs, threshold):
    states = []
    signs = []
    for i in range(0, 2**N):
        val = gs[i][0].real
        if abs(val) > threshold:
            states.append(i)
            signs.append(-1 + 2 * int(val > 0))
    return states, signs

def find_sign_deviations(states, signs, pred_signs):
    dev_forward = []
    for state, true, calc in zip(states, signs, pred_signs):
        if true != calc:
            dev_forward.append(state)
    
    dev_rev = []
    for state, true, calc in zip(states, signs, pred_signs):
        if true != -calc:
            dev_rev.append(state)
    if len(dev_rev) > len(dev_forward):
        return dev_forward
    return dev_rev

def rotate_right(state,rotations,N):
    return (2**N-1)&(state>>rotations|state<<(N-rotations))

def find_uniques(states, N):
    def found(uniques, state, N):
            for u in uniques:
                for rots in range(N):
                    if u == rotate_right(state, rots, N):
                        return u
            return False

    uniques = {}
    for state in states:
        f = found(uniques, state, N)
        if isinstance(f, bool):
            uniques[state] = 1
        else:
            uniques[f] += 1
    return uniques