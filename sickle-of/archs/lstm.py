import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_shape, output_shape, units=288, activation='elu', dropout=0.5, window=3):
        super(LSTMModel, self).__init__()

        self.input_size = int(torch.prod(torch.tensor(input_shape[1:])))
        self.output_shape = output_shape  # e.g., (1, 540)
        #self.output_dim = int(torch.prod(torch.tensor(output_shape)))  # = 1 * 540 = 540

        # drop the time‐window dim first, then compute output_dim = 1*540 = 540
        per_timestep_shape = output_shape[1:] if output_shape[0] == window else output_shape
        self.output_dim = 1
        for dim in per_timestep_shape:
            self.output_dim *= dim

        self.lstm1 = nn.LSTM(self.input_size, units, batch_first=True)
        self.lstm2 = nn.LSTM(units, units, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(units, units)
        self.fc2 = nn.Linear(units, units // 2)
        self.fc3 = nn.Linear(units // 2, self.output_dim)

        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        # x: [B, T, C, D] — flatten to [B, T, features]
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)

        # Apply FC layers to each timestep: [B, T, units] → [B, T, output_dim]
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        # Reshape to [B, T, 1, 540]
        return x.view(x.size(0), x.size(1), 1, -1)


def build_model(input_shape, output_shape, units=288, activation='elu', dropout=0.5, lr=0.0003, window=3):
    return LSTMModel(input_shape=input_shape,
                     output_shape=output_shape,
                     units=units,
                     activation=activation,
                     dropout=dropout,
                     window=window)


def get_meta_model(input_shape):
    def meta_model(hp):
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        dropout = hp.Float('dropout_rate', 0.0, 0.5, sampling='linear')
        return build_model(input_shape, units, activation, dropout)
    return meta_model
