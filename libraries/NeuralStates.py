import numpy.random as npr
import random
import torch
from libraries import utils as utils


class SparseStateVector:
    """
    Container class for dictionary (self.values) with keys of integer states
    and values being complex amplitude of psi
    """
    def __init__(self):
        self.values = {}
        self.normalized = False

    def TFIM_multiply(psi, N, J, Gamma):
        """
        Returns new sparse vector representing H|psi>
        """
        prod = SparseStateVector()
        for state in psi.values:
            jtotal = 0
            for site in range(N - 1):
                jtotal += J if ((state >> site) ^ (state >> site + 1)) & 1 else -J
            jtotal += J if ((state >> (N - 1)) ^ (state >> 0)) & 1 else -J 
            prod.values[state] = jtotal * psi.values[state]
        
        for state in psi.values:
            for site in range(N):
                flipped_state = state ^ (1 << site)
                prod.values[flipped_state] = prod.values.get(flipped_state, 0) - Gamma * psi.values[state]
        return prod

    def inner_product(v1, v2):
        """
        Returns <v1|v2> for two SparseStateVectors
        """
        prod = 0
        for s in v1.values:
            if s in v2.values:
                prod += torch.conj(v1.values[s]) * v2.values[s]
        return prod

    def TFIM_expectation_from_sparse(psi, N, J, Gamma):
        """
        Returns <psi|H|psi>/<psi|psi> for SparseStateVector psi
        """
        # do H|psi> then <psi| (H|psi>)
        hpsi = SparseStateVector.TFIM_multiply(psi, N, J, Gamma)
        exp = SparseStateVector.inner_product(psi, hpsi)
        if not psi.normalized:
            mag2 = SparseStateVector.inner_product(psi, psi)
            return (exp / mag2).real
        return exp.real

    def normalize(self):
        """
        Normalizes values
        """
        mag = sum(abs(self.values[s]) ** 2 for s in self.values) ** 0.5
        for s in self.values:
            self.values[s] = self.values[s] / mag
        self.normalized = True

    def to_prob_distribution(self, N):
        """
        Returns 1D list representing real probability distribution
        """
        if not self.normalized:
            mag2 = sum(abs(self.values[s]) ** 2 for s in self.values)
            return [(abs(self.values.get(s, 0)) ** 2 / mag2).item() for s in range(0, 2 ** N)]
        return [abs(self.values.get(s, 0)) ** 2 for s in range(0, 2 ** N)]
    
    def to_dense_vector(self, N):
        """
        Returns 1D list of dense representation
        """
        if not self.normalized:
            mag = sum(abs(self.values[s]) ** 2 for s in self.values) ** 0.5
            return [(self.values.get(s, 0) / mag).item() for s in range(0, 2 ** N)]
        return [self.values.get(s, 0).item() for s in range(0, 2 ** N)]

class UniformNeuralState(SparseStateVector):
    def __init__(self, N, model, output_to_psi, num_samples):
        """
        Initializes sparse vector values

        Args:
            N (int): number of qubits
            model: torch model representing psi(x), which returns complex amplitude given integer state
            output_to_psi (function): takes in output of model to compute complex amplitude
            num_samples (int): number of unique integer samples to take
        """
        super().__init__()
        self.samples = num_samples
        self.list = []
        self.nn_output = {}
        def psi(x):
            tens = torch.tensor([utils.generate_state_array(x, N)], dtype = torch.float32)
            nn_output = model(tens)
            return output_to_psi(nn_output)[0], nn_output
        if num_samples >= 2 ** N:
            for state in range(2 ** N):
                self.values[state] = psi(state)
                self.list.append(state)
                self.values[state], self.nn_output[state] = psi(state)
        else:
            sampled_states = set()
            for _ in range(num_samples):
                x = random.getrandbits(N)
                while x in sampled_states:
                    x = random.getrandbits(N)
                sampled_states.add(x)
                self.list.append(x)
            for state in sampled_states:
                self.values[state], self.nn_output[state] = psi(state)

class MHNeuralState(SparseStateVector):
    def __init__(self, N, model, output_to_psi, x_func, x0, num_samples, burnin = 0, lag = 0, chains = 1):
        """
        Initializes distribution of samples and vector values

        Args:
            N (int): number of qubits
            model: torch model representing psi(x), which returns complex amplitude given integer state
            output_to_psi (function): takes in output of model to compute complex amplitude
            x_func (function): takes in state x and generates proposal x*
            x0 (int): integer state to begin sampling
            num_samples (int): number of proposal x* generated
            burnin (int): number of samples to throw away before accepting first sample
            lag (int): number of samples to throw away in-between accepting samples
        """
        super().__init__()
        self.distribution = {}
        self.samples = num_samples
        self.list = []
        self.nn_output = {}

        self.N = N
        self.model = model
        self.output_to_psi = output_to_psi
        self.x_func = x_func
        self.x0 = x0
        self.num_samples = num_samples
        self.burnin = burnin
        self.lag = lag
        self.chains = chains # note any of these could possibly be modified by a client
        
        num_uniform = burnin * chains + num_samples * (lag + 1)
        self.rand_uniform = torch.rand(num_uniform)
        self.index = 0
        # single chain uses arbitrary x_func, multi chain only allows bitflip for time optimization
        if chains == 1 and isinstance(x0, int):
            self._init_single_chain()
        elif len(x0) == chains:
            self._init_multi_chain()
        else:
            del self.index
            raise Exception('invalid initial values or number of chains')
        del self.index
        
    def _init_single_chain(self):
        def psi(x):
            tens = torch.tensor([utils.generate_state_array(x, self.N)], dtype = torch.float32)
            nn_output = self.model(tens)
            return self.output_to_psi(nn_output)[0], nn_output[0]

        x = self.x0

        def run_single_sample(modify = False):
            nonlocal x
            nonlocal psi_val
            new_x = self.x_func(x)
            if new_x in self.values: new_psi_val, new_nn_val = self.values[new_x], self.nn_output[new_x]
            else: new_psi_val, new_nn_val = psi(new_x)
            ratio = abs(new_psi_val) ** 2 / abs(psi_val) ** 2
            if ratio > 1 or ratio > self.rand_uniform[self.index]:
                if modify:
                    self.distribution[new_x] = self.distribution.get(new_x, 0) + 1
                    self.list.append(new_x)
                x = new_x 
                psi_val = new_psi_val 
            elif modify: 
                self.distribution[x] = self.distribution.get(x, 0) + 1
                self.list.append(x)
            self.values[new_x] = new_psi_val
            self.nn_output[new_x] = new_nn_val
            self.index += 1

        psi_val, nn_val = psi(x)
        self.values[x] = psi_val
        self.nn_output[x] = nn_val
        for _ in range(self.burnin):
            run_single_sample(modify = False)
        for _ in range(self.num_samples):
            for _ in range(self.lag):
                run_single_sample(modify = False)
            run_single_sample(modify = True)

    def _init_multi_chain(self):
        def psi(xs):
            tens = utils.generate_input_samples_torch(xs, self.N)
            nn_output = self.model(tens)
            res = self.output_to_psi(nn_output)
            return res, nn_output

        xs = self.x0[:]
        psi_vals, nn_vals = psi(xs)
        for i, x in enumerate(xs):
            self.values[x] = psi_vals[i]
            self.nn_output[x] = nn_vals[i]
        
        for _ in range(self.burnin):
            self._run_single_chained_sample(xs, psi_vals, psi, modify = False)
        num_iters = self.num_samples // self.chains
        remainder = self.num_samples % self.chains
        for c in range(num_iters):
            for _ in range(self.lag):
                self._run_single_chained_sample(xs, psi_vals, psi, modify = False)
            self._run_single_chained_sample(xs, psi_vals, psi, modify = True)
        if remainder != 0:
            xs = xs[:remainder]
            psi_vals = psi(xs)[0]
            self._run_single_chained_sample(xs, psi_vals, psi, modify = True)
    
    def _run_single_chained_sample(self, xs, psis, psi_function, modify = False):
        # only allows bitflipping as x*=f(x) function
        new_xs = utils.bitflip_batch(xs, self.N, 1)
        new_psi_vals, new_nn_vals = psi_function(new_xs)
        ratios = torch.abs(new_psi_vals) ** 2 / torch.abs(psis) ** 2

        accept_mask = (ratios > 1) | (ratios > self.rand_uniform[self.index : self.index + len(xs)])
        xs[accept_mask] = new_xs[accept_mask]
        psis[accept_mask] = new_psi_vals[accept_mask]

        if modify:
            accepted = new_xs[accept_mask]
            rejected = xs[~accept_mask]

            for x in accepted.tolist():
                self.distribution[x] = self.distribution.get(x, 0) + 1
                self.list.append(x)

            for x in rejected.tolist():
                self.distribution[x] = self.distribution.get(x, 0) + 1
                self.list.append(x)

        for x, val, nn in zip(new_xs.tolist(), new_psi_vals.tolist(), new_nn_vals.tolist()):
            self.values[x] = val
            self.nn_output[x] = nn
        
        self.index += len(xs)
        