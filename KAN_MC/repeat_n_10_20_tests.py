from libraries import lib
from libraries.NeuralStates import *
import qutip as qt
import torch.nn as nn 
import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import *
import pickle
import time

def generate_eloc_distr(sampled_vector, N, J, Gamma, model):
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
    
    eloc_values = {}
    for basis_state in sampled_vector.distribution:
        eloc = 0
        output = model_to_output(basis_state)
        for adjacency in lib.generate_adjacencies(basis_state, N):
            output_prime = model_to_output(adjacency)
            eloc += lib.calc_H_elem(N, J, Gamma, basis_state, adjacency) * torch.exp(output_prime[0] - output[0] + 1.j * (output_prime[1] - output[1]))
        eloc_values[basis_state] = eloc
    return eloc_values

def generate_input_samples(N, samples):
    return torch.tensor([lib.generate_state_array(x, N) for x in samples]).to(torch.float32)

def log_amp_phase(nn_output):
    return torch.exp(nn_output[:, 0] + 1.j * nn_output[:, 1])

def bitflip_x(x, N, flips):
    new_x = x
    for _ in range(flips):
        new_x = x ^ (1 << npr.randint(0, N))
    return new_x

npr.seed(0)
torch.manual_seed(0)

def get_file(mode='a'):
    return None #open('data/n_10_20_tests_log.txt', mode)

f = get_file('w')
f.close()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_data = {} # keyed by (N, h, trial)

n_values = [10, 20]; h_values = [0.1, 1, 10]
J = 1
num_epochs = 200
num_trials = 3
num_samples = 256
data_rate = 1

orig = time.time()

for N in n_values:
    for Gamma in h_values:
        true_gse = lib.ground_state_energy(Gamma, N)
        print(f'true GSE for N={N}, h={Gamma} is: {true_gse} - time: {time.time() - orig}')
        f = get_file()
        f.write(f'true GSE for N={N}, h={Gamma} is: {true_gse} \n')
        for trial in range(num_trials):
            print(f'trial {trial} - time: {time.time() - orig}')
            f.write(f'trial {trial}\n')
            kan_model = KAN(width=[N, N, 2], device=device, seed=trial, auto_save=False);

            start = time.time()
            epochs = []
            energy_data = []
            optimizer = torch.optim.Adam(kan_model.parameters(), lr=0.01)

            for epoch in range(num_epochs):
                mh_state = MHNeuralState(N, kan_model, log_amp_phase, lambda x : bitflip_x(x, N, 1), 0, num_samples)
                eloc_distr = generate_eloc_distr(mh_state, N, J, Gamma, kan_model)
                eloc_list = torch.tensor([eloc_distr[x] for x in mh_state.list])
                psi = kan_model(generate_input_samples(N, mh_state.list))
                log_amp = psi[:, 0]; phase = psi[:, 1]
                eloc_r = eloc_list.real
                eloc_i = eloc_list.imag
                energy = eloc_r.mean()
                loss = ((eloc_r - eloc_r.mean()).detach() * log_amp + (eloc_i - eloc_i.mean()).detach() * phase).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % data_rate == 0:
                    energy_data.append(energy)
                    epochs.append(epoch)
                if epoch % 10 == 0:
                    f.write(f'epoch: {epoch}, energy: {energy}\n')
            all_data[(N, Gamma, trial)] = energy_data
            f.write(f'final: {energy_data[-1]}\n')
            f.write(f'minimum: {min(energy_data)}\n')
            f.write(f'N={N}, h={Gamma}, trial={trial} took {time.time() - start} seconds\n')

        for i in range(num_trials):
            plt.plot(epochs, all_data[(N, Gamma, i)], label = f'trial {i}')
        plt.plot([0, epochs[-1]], 2*[true_gse], label = 'true')
        plt.legend(loc='best')
        plt.title(f'training loss with N={N}, h={Gamma}, {num_samples} samples')
        plt.savefig(f'plots/loss_KAN_N_{N}_h_10_{round(np.log10(Gamma), 2)}_samples_{num_samples}.png')
        plt.clf()

        for i in range(num_trials):
            err = (np.array(all_data[(N, Gamma, i)]) - true_gse) ** 2
            plt.plot(epochs, err, label = f'trial {i}')
        plt.legend(loc='best')
        plt.title(f'training error with N={N}, h={Gamma}, {num_samples} samples')
        plt.yscale('log')
        plt.savefig(f'plots/error_KAN_N_{N}_h_10_{round(np.log10(Gamma), 2)}_samples_{num_samples}.png')
        plt.clf()
        f.close()

end = time.time()
f = get_file()
f.write(f'total time: {end - orig}\n')
print(f'total time: {end - orig}')
with open('data/n_10_20_tests_data.p', 'wb') as fp:
    pickle.dump(all_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
