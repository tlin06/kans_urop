import numpy.random as npr
import torch
import lib 

class SparseStateVector:
    def __init__(self):
        self.magnitudes = {}
        self.values = {}

    def TFIM_multiply(psi, N, J, Gamma):
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
        prod = 0
        for s in v1.values:
            if s in v2.values:
                prod += torch.conj(v1.values[s]) * v2.values[s]
        return prod

    def TFIM_expectation_from_sparse(psi, N, J, Gamma):
        # do H|psi> then <psi| (H|psi>)
        hpsi = SparseStateVector.TFIM_multiply(psi, N, J, Gamma)
        exp = SparseStateVector.inner_product(psi, hpsi)
        mag = SparseStateVector.inner_product(psi, psi)
        return (exp / mag).real

class MHNeuralState(SparseStateVector):
    def __init__(self, N, model, output_to_psi, x_func, x0, num_samples, burnin = 0, lag = 0):
        super().__init__()
        self.distribution = {}
        self.samples = num_samples
        def psi(x):
            tens = torch.tensor([lib.generate_state_array(x, N)], dtype = torch.float32)
            return output_to_psi(model(tens))[0]
        num_uniform = burnin + num_samples * (lag + 1)
        rand_uniform = npr.uniform(0, 1, num_uniform)
        index = 0
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
        for state in self.distribution:
            self.magnitudes[state] = self.distribution[state] / self.samples

    def distribution_to_list(self):
        return [state for state in self.distribution for _ in range(self.distribution[state])]
