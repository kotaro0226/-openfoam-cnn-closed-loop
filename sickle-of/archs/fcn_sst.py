import torch
import torch.nn as nn
import torch.optim as optim

class FCN_SST(nn.Module):
    def __init__(self, input_shape, output_shape, units1=2000, units2=8000, units3=2000, activation='elu'):
        super(FCN_SST, self).__init__()
        units0 = int(input_shape[-1])
        units4 = int(output_shape[-1])
        self.fc1 = nn.Linear(units0, units1)
        self.fc2 = nn.Linear(units1, units2)
        self.fc3 = nn.Linear(units2, units3)
        self.out = nn.Linear(units3, units4)
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.out(x)
        return x

def build_model(input_shape, output_shape, units1=2000, units2=8000, units3=2000, activation='elu', window=1):
    model = FCN_SST(input_shape, output_shape, units1, units2, units3, activation)
    return model

def get_meta_model(input_shape):
    def meta_model(hp):
        units1 = hp.Int('units1', min_value=2, max_value=8000, step=800)
        units2 = hp.Int('units2', min_value=2, max_value=8000, step=800)
        units3 = hp.Int('units2', min_value=2, max_value=8000, step=800)
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        return build_model(input_shape, output_shape, units1, units2, units3, activation)
    return meta_model
