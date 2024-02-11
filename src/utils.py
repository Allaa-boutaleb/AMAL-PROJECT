import torch
import numpy as np
import tqdm
import math
from math import ceil
import sys


def isometry_gap(X):
    """
    Computes the isometry gap of a matrix X.
    
    Args:
        X (torch.Tensor): Input matrix.
    
    Returns:
        torch.Tensor: Isometry gap of X.
    """
    G = X @ X.t()
    G = G.detach()
    eigenvalues = torch.linalg.eigvalsh(G)
    return -torch.log(eigenvalues).mean() + torch.log(eigenvalues.mean())



def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch over the provided data loader.
    
    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (callable): Loss function.
        device (torch.device): Device to run the training on.
    
    Returns:
        tuple: Tuple containing average training loss and accuracy.
    """
    model.train()  # Set the model to training mode
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm.tqdm(loader, leave=False, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
    
    # Check for NaN loss and exit if found
    if math.isnan(running_loss):
        print("Training loss is NaN.", file=sys.stderr)
        sys.exit(1)
    
    return running_loss / len(loader), correct / total


def test_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model for one epoch over the provided data loader.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for test data.
        criterion (callable): Loss function.
        device (torch.device): Device to run the evaluation on.
    
    Returns:
        tuple: Tuple containing average test loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():  # Disable gradient computation
        pbar = tqdm.tqdm(loader, leave=False, desc="Testing")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
    
    # Check for NaN loss and exit if found
    if math.isnan(running_loss):
        print("Testing loss is NaN.", file=sys.stderr)
        sys.exit(1)
    
    return running_loss / len(loader), correct / total


################################################################################################

def measure_layer(model, l, epoch):
    """
    Measures various properties of a specific layer in the model.
    
    Args:
        model (torch.nn.Module): The model.
        l (int): Layer index.
        epoch (int): Current epoch number.
    
    Returns:
        dict: A dictionary containing measurements of the layer.
    """
    w = model.layers[f'fc_{l}'].weight
    w_grad = w.grad
    w_grad_fro = w_grad.norm('fro').item()
    w_ig = isometry_gap(w).item()
    fc_ig = isometry_gap(model.hiddens[f'fc_{l}']).item()
    
    if l < model.num_layers:
        act_ig = isometry_gap(model.hiddens[f'act_{l}']).item()
        norm_ig = isometry_gap(model.hiddens[f'norm_{l}']).item()
    else:
        act_ig = np.nan  # Not applicable for the final layer
        norm_ig = np.nan  # Not applicable for the final layer
    
    return {
        'layer': l,
        'epoch': epoch,
        'weight_isogap': w_ig,
        'fc_isogap': fc_ig,
        'act_isogap': act_ig,
        'norm_isogap': norm_ig,
        'grad_fro_norm': w_grad_fro,
    }
    
    
def get_measurements(model, inputs, labels, criterion, epoch, device):
    """
    Performs a forward and backward pass to gather measurements on model's layers.
    
    Args:
        model (torch.nn.Module): The neural network model.
        inputs (torch.Tensor): Batch of input data.
        labels (torch.Tensor): Batch of labels.
        criterion (Callable): Loss function.
        epoch (int): Current epoch number.
        device (torch.device): Device to run computations on.
    
    Returns:
        list[dict]: Measurements for each layer of the model.
    """
    model.train()
    inputs, labels = inputs.to(device), labels.to(device)
    model.set_save_hidden(True)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    measurements = []
    for l in tqdm.tqdm(range(model.num_layers + 1), desc="Layer measurements"):
        layer_measurements = measure_layer(model, l, epoch)
        measurements.append(layer_measurements)

    model.zero_grad(set_to_none=True)  # Clean gradients
    model.set_save_hidden(False)  # Disable saving hidden states
    return measurements

################################################################################################

def dataset_to_tensors(dataset, indices=None, device='cuda'):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).to(device)
    return x, y


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch sampling
    and adaptive augmentation on GPU."""

    def __init__(
        self,
        x,
        y,
        batch_size=500,
        shuffle=False,
    ):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.device = x.device
        self.n_data = y.size(0)
        self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data, device=self.device)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        x, y = self.x[start:end], self.y[start:end]
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return self.n_batches

    def attach(self):
        self._detach = False
        return self

    def detach(self):
        self._detach = True
        return self

    @property
    def dataset(self):
        return DatasetDummy(self.n_data)


class DatasetDummy:
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return int(self.N)
