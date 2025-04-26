from libraries import lib
from libraries.NeuralStates import *
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import *
import time

import warnings
warnings.filterwarnings('ignore')

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
    return open('data/timing_6_tests_log.txt', mode)

f = get_file('w')
f.close()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_data = pd.DataFrame({'N':[], 'h':[], 'trial':[], 'epoch':[], 'energy':[], 'epoch time':[]})
all_data = all_data.set_index(['N', 'h', 'trial', 'epoch'])

time_keys = ['initiate KAN', 'sampling', "generate eloc distr", 'convert to tensor', 
             'generate input samples', 'forward pass', 'calc loss', 
             'loss backward', 'optimizer step']

n_values = [6]; h_values = [0.1, 1, 10]
J = 1
num_epochs = 200
num_trials = 10
num_samples = 128
data_rate = 1

orig = time.time()

for N in n_values:
    for Gamma in h_values:

        # f.write(f'processing N={N}, h={Gamma}')
        times = {key:0 for key in time_keys}

        f = get_file()
        for trial in range(num_trials):
            start_trial = time.time()
            # print(f'processing N={N}, h={Gamma}, trial {trial} - time: {time.time() - orig}')
            # f.write(f'trial {trial}\n')

            start = time.time()
            kan_model = KAN(width=[N, N, 2], device=device, seed=trial, auto_save=False);

            optimizer = torch.optim.Adam(kan_model.parameters(), lr=0.01)
            times['initiate KAN'] += time.time() - start

            for epoch in range(num_epochs):
                start_epoch = time.time()

                start = time.time()
                mh_state = MHNeuralState(N, kan_model, log_amp_phase, lambda x : bitflip_x(x, N, 1), 0, num_samples)
                times['sampling'] += time.time() - start

                start = time.time()
                eloc_distr = generate_eloc_distr(mh_state, N, J, Gamma, kan_model)
                times['generate eloc distr'] += time.time() - start

                start = time.time()
                eloc_list = torch.tensor([eloc_distr[x] for x in mh_state.list])
                times['convert to tensor'] += time.time() - start

                start = time.time()
                input_samples = generate_input_samples(N, mh_state.list)
                times['generate input samples'] += time.time() - start

                start = time.time()
                psi = kan_model(input_samples)
                times['forward pass'] += time.time() - start

                start = time.time()
                log_amp = psi[:, 0]; phase = psi[:, 1]
                eloc_r = eloc_list.real
                eloc_i = eloc_list.imag
                energy = eloc_r.mean()
                loss = ((eloc_r - eloc_r.mean()).detach() * log_amp + (eloc_i - eloc_i.mean()).detach() * phase).mean()
                times['calc loss'] += time.time() - start

                start = time.time()
                optimizer.zero_grad()
                loss.backward()
                times['loss backward'] += time.time() - start

                start = time.time()
                optimizer.step()
                times['optimizer step'] += time.time() - start

                epoch_time = time.time() - start_epoch
                all_data.loc[(N, Gamma, trial, epoch), ['energy', 'epoch time']] = energy.item(), epoch_time

            trial_time = time.time() - start_trial
            f.write(f'processing N={N}, h={Gamma}, trial {trial} took {trial_time} seconds\n')
            print(f'processing N={N}, h={Gamma}, trial {trial} took {trial_time} seconds')
        
        total_h_time = sum(v for v in times.values())
        f.write(f'N={N}, h={Gamma} took {total_h_time} seconds\n')
        f.write(f'the distribution was: \n')
        for key in time_keys:
            f.write(f'{key} took {times[key]} seconds\n')
        f.write('\n')
        f.close()

end = time.time()
f = get_file()
f.write(f'total runtime: {end - orig}\n')
print(f'total runtime: {end - orig}')
all_data.to_csv('data/KAN_n_6_data.csv')
f.close()
