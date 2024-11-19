import numpy.random as npr
import random
import torch
import lib 

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
        def __init__(self, N, model, output_to_psi, num_samples, informed = False):
            """
            Initializes sparse vector values

            Args:
                N (int): number of qubits
                model: torch model representing psi(x), which returns complex amplitude given integer state
                output_to_psi (function): takes in output of model to compute complex amplitude
                num_samples (int): number of unique integer samples to take
                informed (bool): whether to guarantee sample first and last states
            """
            super().__init__()
            self.samples = num_samples
            def psi(x):
                tens = torch.tensor([lib.generate_state_array(x, N)], dtype = torch.float32)
                return output_to_psi(model(tens))[0]
            if num_samples >= 2 ** N:
                for state in range(2 ** N):
                    self.values[state] = psi(state)
            else:
                sampled_states = set()
                if informed: 
                    sampled_states.add(0)
                    sampled_states.add(2 ** N - 1)
                    num_samples = num_samples - 2
                for _ in range(num_samples):
                    x = random.getrandbits(N)
                    while x in sampled_states:
                        x = random.getrandbits(N)
                    sampled_states.add(x)
                for state in sampled_states:
                    self.values[state] = psi(state)

class MHNeuralState(SparseStateVector):
    def __init__(self, N, model, output_to_psi, x_func, x0, num_samples, burnin = 0, lag = 0, informed = False):
        """
        Initializes distribution of samples and vector values

        Args:
            N (int): number of qubits
            model: torch model representing psi(x), which returns complex amplitude given integer state
            output_to_psi (function): takes in output of model to compute complex amplitude
            x_func (function): takes in state x and generates proposal x*
            x0 (int): intger state to begin sampling
            num_samples (int): number of proposal x* generated
            burnin (int): number of samples to throw away before accepting first sample
            lag (int): number of samples to throw away in-between accepting samples
            informed (bool): whether to guarantee sample first and last states
        """
        # uses arbitrary x_func for MH sampling
        super().__init__()
        self.distribution = {}
        self.samples = num_samples
        def psi(x):
            tens = torch.tensor([lib.generate_state_array(x, N)], dtype = torch.float32)
            return output_to_psi(model(tens))[0]
        num_uniform = burnin + num_samples * (lag + 1)
        rand_uniform = npr.uniform(0, 1, num_uniform)
        index = 0
        if informed: 
            self.values[0] = psi(0)
            self.values[2 ** N - 1] = psi(2 ** N - 1)
            self.distribution[0] = 1
            self.distribution[2 ** N - 1] = 1
            num_samples = num_samples - 2
        x = x0
        val = psi(x)
        for _ in range(burnin):
            new_x = x_func(x)
            new_val = self.values[new_x] if new_x in self.values else psi(new_x)
            ratio = abs(new_val) ** 2 / abs(val) ** 2
            if ratio > 1 or ratio > rand_uniform[index]:
                x = new_x
                val = new_val
            index += 1
        for _ in range(num_samples):
            for _ in range(lag):
                new_x = x_func(x)
                new_val = self.values[new_x] if new_x in self.values else psi(new_x)
                ratio = abs(new_val) ** 2 / abs(val) ** 2
                if ratio > 1 or ratio > rand_uniform[index]:
                    x = new_x
                    val = new_val
                index += 1
            new_x = x_func(x)
            new_val = self.values[new_x] if new_x in self.values else psi(new_x)
            ratio = abs(new_val) ** 2 / abs(val) ** 2
            if ratio > 1 or ratio > rand_uniform[index]:
                self.distribution[new_x] = self.distribution.get(new_x, 0) + 1
                self.values[new_x] = new_val
                x = new_x 
                val = new_val 
            else: 
                self.distribution[x] = self.distribution.get(x, 0) + 1
                self.values[new_x] = new_val
            index += 1

    def distribution_to_list(self):
        """
        Returns list of sampled states with repetition
        """
        return [state for state in self.distribution for _ in range(self.distribution[state])]
