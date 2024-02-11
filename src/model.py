import torch
from torch import nn
from math import sqrt

class MLPWithBatchNorm(nn.Module):
    """
    A multilayer perceptron (MLP) with customizable normalization and activation layers.
    """
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, norm_type, mean_reduction, activation, save_hidden, exponent, order='norm_act', force_factor=None, bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.save_hidden = save_hidden
        self.exponent = exponent
        self.order = order
        self.hiddens = {} if save_hidden else None

        if order not in ['act_norm', 'norm_act']:
            raise ValueError("Invalid order specified.")

        # Building layers
        self.layers = self.build_layers(num_layers, hidden_dim, norm_type, mean_reduction, activation, force_factor, bias)
        self.initialized = False

    def build_layers(self, num_layers, hidden_dim, norm_type, mean_reduction, activation, force_factor, bias):
        """
        Builds the layers of the MLP.

        Args:
            num_layers (int): Number of layers in the MLP.
            hidden_dim (int): Dimension of the hidden layers.
            norm_type (str): Type of normalization.
            mean_reduction (bool): If True, normalization includes mean reduction.
            activation (callable): Activation function.
            force_factor (float, optional): Force factor for custom normalization.
            bias (bool): If True, adds a learnable bias to the layers.

        Returns:
            nn.ModuleDict: A dictionary of layers.
        """
        layers = nn.ModuleDict()

        # Input layer setup
        layers['fc_0'] = nn.Linear(self.input_dim, hidden_dim, bias=bias)
        layers['norm_0'] = self.create_norm_layer(hidden_dim, norm_type, mean_reduction, force_factor)
        layers['act_0'] = activation()

        # Hidden layers setup
        for l in range(1, num_layers):
            layers[f'fc_{l}'] = nn.Linear(hidden_dim, hidden_dim, bias=bias)
            layers[f'norm_{l}'] = self.create_norm_layer(hidden_dim, norm_type, mean_reduction, force_factor)
            layers[f'act_{l}'] = activation()

        # Output layer setup
        layers[f'fc_{num_layers}'] = nn.Linear(hidden_dim, self.output_dim, bias=bias)

        return layers

    def create_norm_layer(self, dim, norm_type, mean_reduction, force_factor):
        """
        Creates a normalization layer based on the specified parameters.

        Args:
            dim (int): Dimension of the layer.
            norm_type (str): Type of normalization ('torch_bn', 'bn', 'ln', 'id').
            mean_reduction (bool): If True, includes mean reduction.
            force_factor (float, optional): Custom scaling factor.

        Returns:
            nn.Module: The normalization layer.
        """
        if norm_type == 'torch_bn':
            return nn.BatchNorm1d(dim)
        else:
            return CustomNormalization(norm_type, mean_reduction, force_factor=force_factor)

    def forward(self, x):
        assert self.initialized, "Model not initialized."
        x = x.view(-1, self.input_dim)

        for l in range(self.num_layers):
            layer_gain = ((l + 1) ** self.exponent)
            x = self.layers[f'fc_{l}'](x)
            if self.save_hidden:
                self.hiddens[f'fc_{l}'] = x.detach()

            if self.order == 'norm_act':
                x = self.apply_norm_act(x, l, layer_gain)
            elif self.order == 'act_norm':
                x = self.apply_act_norm(x, l, layer_gain)

        x = self.layers[f'fc_{self.num_layers}'](x)
        if self.save_hidden:
            self.hiddens[f'fc_{self.num_layers}'] = x.detach()
        return x

    def apply_norm_act(self, x, l, layer_gain):
        """
        Applies normalization followed by activation to the input tensor.
        
        Args:
            x (Tensor): Input tensor.
            l (int): Layer index.
            layer_gain (float): Gain factor for the layer.
        
        Returns:
            Tensor: The processed tensor.
        """
        x = self.layers[f'norm_{l}'](x)
        if self.save_hidden:
            self.hiddens[f'norm_{l}'] = x.detach()
        x = self.layers[f'act_{l}'](x * layer_gain)
        if self.save_hidden:
            self.hiddens[f'act_{l}'] = x.detach()
        return x

    def apply_act_norm(self, x, l, layer_gain):
        """
        Applies activation followed by normalization to the input tensor.
        
        Args:
            x (Tensor): Input tensor.
            l (int): Layer index.
            layer_gain (float): Gain factor for the layer.
        
        Returns:
            Tensor: The processed tensor.
        """
        x = self.layers[f'act_{l}'](x * layer_gain)
        if self.save_hidden:
            self.hiddens[f'act_{l}'] = x.detach()
        x = self.layers[f'norm_{l}'](x)
        if self.save_hidden:
            self.hiddens[f'norm_{l}'] = x.detach()
        return x


    def reset_parameters(self, init_type, gain=1.0):
        """
        Resets the parameters of the network using the specified initialization scheme.
        
        Args:
            init_type (str): Initialization type ('xavier_normal', 'orthogonal').
            gain (float): Gain factor for initialization.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight, gain=gain)
                else:
                    raise ValueError("Unsupported initialization type.")
        self.initialized = True
        
        
class SinAct(nn.Module):
    """
    A custom sinusoidal activation function module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Applies the sine function to the input tensor.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Transformed tensor.
        """
        return torch.sin(x)



class CustomNormalization(nn.Module):
    """
    A custom normalization module that supports batch normalization, layer normalization, and identity operations.
    """
    def __init__(self, norm_type, mean_reduction, force_factor=None):
        super().__init__()
        self.mean_reduction = mean_reduction
        self.norm_type = norm_type
        self.force_factor = force_factor

        if norm_type == 'bn':
            self.dim = 0
        elif norm_type == 'ln':
            self.dim = 1
        elif norm_type == 'id':
            self.dim = -1
        else:
            raise ValueError("Unsupported normalization type.")

    def forward(self, X):
        if self.dim == -1:
            return X

        if self.mean_reduction:
            X = X - X.mean(dim=self.dim, keepdim=True)

        norm = X.norm(dim=self.dim, keepdim=True)
        factor = sqrt(X.shape[self.dim]) if self.force_factor is None else self.force_factor
        X = X / (norm / factor)
        return X