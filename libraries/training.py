import math
import qutip as qt
import torch
import libraries.utils as utils

def model_to_ground_state(N, model, output_to_psi):
    """
    Args:
        model: trained torch model
        output_to_psi: function that takes a (2^N, 2) tensor output 
        from a torch network and returns a 2^N size tensor

    Returns:
        Returns normalized Qobj state derived from model
    """
    input = utils.generate_input_torch(N)
    pred = model(input)
    pred_gs = output_to_psi(pred)
    mag = math.sqrt(sum(abs(n) ** 2 for n in pred_gs))
    pred_gs = qt.Qobj(pred_gs.data / mag)
    return pred_gs

def train_model_to_gs(model, generate_y_pred, loss_fn, num_epochs, optim = torch.optim.SGD, learning_rate = 2, data_rate = 50):
    """
    Trains model to find the ground state

    Args:
        model: torch model to be trained
        generate_y_pred (func): that takes in model and outputs psi
        loss_fn (func): takes in psi and outputs loss
        num_epochs (int): steps to train for
        optim (torch.optim.Optimizer): optimizer to use
        learning_rate (float): learning rate to run model with
        data_rate (int): collects loss data every data_rate epochs
    
    Returns:
        tuple of epochs at which loss was collected and loss at those epochs
    """
    epochs = []
    loss_data = []
    optimizer = optim(model.parameters(), lr = learning_rate)
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