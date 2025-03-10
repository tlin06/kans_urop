from libraries import lib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from libraries.NeuralStates import *
import time
import pickle

def set_gradients(sampled_vector: MHNeuralState, N, J, Gamma, model: nn.Sequential): # also only for log(amp), phase
    nn_output_calcs = {}
    def model_to_output(x):
        if x in sampled_vector.nn_output:
            return sampled_vector.nn_output[x]
        if x in nn_output_calcs:
            return nn_output_calcs[x]
        tens = torch.tensor([lib.generate_state_array(x, N)], dtype = torch.float32)
        output = model(tens)[0]
        nn_output_calcs[x] = output
        return output
    def output_to_log(x):
        return x[0] + 1.j * x[1]

    elocs = {}
    re_grad_logs = {}
    im_grad_logs = {}
    params = [p for p in model.parameters()]
    for basis_state in sampled_vector.distribution:
        eloc = 0
        nn_output = sampled_vector.nn_output[basis_state]
        for adjacency in lib.generate_adjacencies(basis_state, N):
            nn_output_prime = model_to_output(adjacency)
            eloc += lib.calc_H_elem(N, J, Gamma, basis_state, adjacency) * torch.exp(output_to_log(nn_output_prime) - output_to_log(nn_output))
        elocs[basis_state] = eloc.detach()
        
        nn_output[0].backward(retain_graph=True)
        real_grads = [p.grad.clone() for p in params]
        for p in params: p.grad = None
        nn_output[1].backward(retain_graph=True)
        imag_grads = [-p.grad.clone() for p in params]
        re_grad_logs[basis_state] = real_grads
        im_grad_logs[basis_state] = imag_grads

    E_eloc = (sum(sampled_vector.distribution[x] * elocs[x] for x in sampled_vector.distribution) / sampled_vector.samples).real

    for i in range(len(params)):
        E_grad_times_loc = sum((re_grad_logs[x][i] * elocs[x].real - im_grad_logs[x][i] * elocs[x].imag) * \
                               sampled_vector.distribution[x] for x in sampled_vector.distribution) / sampled_vector.samples
        E_grad = sum(re_grad_logs[x][i] * sampled_vector.distribution[x] for x in sampled_vector.distribution) / sampled_vector.samples
        params[i].grad = E_grad_times_loc - E_grad * E_eloc

    return E_eloc

def set_gradients_exact(sampled_vector: MHNeuralState, N, J, Gamma, model: nn.Sequential): # also only for log(amp), phase
    nn_output_calcs = {}
    def model_to_output(x):
        if x in sampled_vector.nn_output:
            return sampled_vector.nn_output[x]
        if x in nn_output_calcs:
            return nn_output_calcs[x]
        tens = torch.tensor([lib.generate_state_array(x, N)], dtype = torch.float32)
        output = model(tens)[0]
        nn_output_calcs[x] = output
        return output
    def output_to_log(x):
        return x[0] + 1.j * x[1]

    elocs = {}
    re_grad_logs = {}
    im_grad_logs = {}
    amp = 0
    params = [p for p in model.parameters()]
    for basis_state in sampled_vector.values:
        eloc = 0
        nn_output = sampled_vector.nn_output[basis_state]
        for adjacency in lib.generate_adjacencies(basis_state, N):
            nn_output_prime = model_to_output(adjacency)
            eloc += lib.calc_H_elem(N, J, Gamma, basis_state, adjacency) * torch.exp(output_to_log(nn_output_prime) - output_to_log(nn_output))
        elocs[basis_state] = eloc.detach()
        
        nn_output[0].backward(retain_graph=True)
        real_grads = [p.grad.clone() for p in params]
        for p in params: p.grad = None
        nn_output[1].backward(retain_graph=True)
        imag_grads = [-p.grad.clone() for p in params]
        re_grad_logs[basis_state] = real_grads
        im_grad_logs[basis_state] = imag_grads

        amp += abs(sampled_vector.values[basis_state].detach()) ** 2

    E_eloc = (sum(abs(sampled_vector.values[x].detach())** 2 * elocs[x] for x in sampled_vector.values) / amp).real

    for i in range(len(params)):
        E_grad_times_loc = sum((re_grad_logs[x][i] * elocs[x].real - im_grad_logs[x][i] * elocs[x].imag) * \
                                abs(sampled_vector.values[x].detach()) ** 2 for x in sampled_vector.values) / amp
        E_grad = sum(re_grad_logs[x][i] * abs(sampled_vector.values[x].detach()) ** 2 for x in sampled_vector.values) / amp
        params[i].grad = E_grad_times_loc - E_grad * E_eloc

    return E_eloc

def log_amp_phase(nn_output):
    return torch.exp(nn_output[:, 0] + 1.j * nn_output[:, 1])
def bitflip_x(x, N, flips):
    new_x = x
    for _ in range(flips):
        new_x = x ^ (1 << npr.randint(0, N))
    return new_x

npr.seed(0)
torch.manual_seed(0)
f = open('data/grad_comparison_log.txt', 'w')
all_data = {} # keyed by (N, h, optim_name, method)


n_values = [6, 10, 20]; J = 1; h_values = [0.1, 1, 10]
methods = ['auto pytorch', 'manual with list', 'manual with values']
optimizers = ['SGD', 'Adam']
num_epochs = 300
data_rate = 1
num_samples = 256

orig = time.time()

for N in n_values:
    for Gamma in h_values:
        true_gse = lib.ground_state_energy(Gamma, N)
        print(f'true GSE for N={N}, h={Gamma} is: {true_gse} - time: {time.time() - orig}')
        f.write(f'true GSE for N={N}, h={Gamma} is: {true_gse} \n')
        for optim_name in optimizers:
            for method in methods:
                # create model
                layers = []
                layers.append(nn.Linear(N, 32))
                for _ in range(2):
                    layers.append(nn.Linear(32, 32))
                    layers.append(nn.SELU())
                layers.append(nn.Linear(32, 2))
                model = nn.Sequential(*layers)
                epochs = [n for n in range(num_epochs)]

                start = time.time()
                energy_data = []
                f.write(method + ' ' + optim_name + '\n')
                if optim_name == 'SGD': optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
                else: optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
                for epoch in range(num_epochs):
                    mh_state = MHNeuralState(N, model, log_amp_phase, lambda x: bitflip_x(x, N, 1), 2 ** (N - 1), num_samples)
                    optimizer.zero_grad()
                    if method == 'auto pytorch': 
                        energy = lib.TFIM_expectation_using_locals(mh_state, N, J, Gamma, model, log_amp_phase)
                        energy.backward()
                    elif method == 'manual with list':
                        energy = set_gradients(mh_state, N, J, Gamma, model)
                    else:
                        energy = set_gradients_exact(mh_state, N, J, Gamma, model)
                    optimizer.step()
                    if epoch % data_rate == 0:
                        energy_data.append(energy.item().real)
                    if epoch % 20 == 0:
                        f.write(f'epoch: {epoch}, energy: {energy}\n')
                all_data[(N, Gamma, optim_name, method)] = energy_data
                f.write(f'final: {energy_data[-1]}\n')
                f.write(f'minimum: {min(energy_data)}\n')
                f.write(f'method {method} with optimizer={optim_name} took {time.time() - start} seconds\n')

            for i in range(len(methods)):
                plt.plot(epochs, all_data[(N, Gamma, optim_name, methods[i])], label = methods[i])
            plt.plot([0, epochs[-1]], 2*[true_gse], label = 'true')
            plt.legend(loc='best')
            plt.title(f'training loss with N={N}, h={Gamma}, {num_samples} samples, {optim_name}')
            plt.savefig(f'plots/loss_N_{N}_h_10_{round(np.log10(Gamma), 2)}_optim_{optim_name}_samples_{num_samples}.png')
            plt.clf()

            for i in range(len(methods)):
                err = (np.array(all_data[(N, Gamma, optim_name, methods[i])]) - true_gse) ** 2
                plt.plot(epochs, err, label = methods[i])
            plt.legend(loc='best')
            plt.title(f'training error with N={N}, h={Gamma}, {num_samples} samples, {optim_name}')
            plt.savefig(f'plots/error_N_{N}_h_10_{round(np.log10(Gamma), 2)}_optim_{optim_name}_samples_{num_samples}.png')
            plt.clf()
print(f'total time: {time.time() - orig}')
with open('data/grad_comparisons_data.p', 'wb') as fp:
    pickle.dump(all_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
f.close()