import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_heads=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_shape[-1]
        self.output_dim = output_shape[-1]

        # Encoder
        self.embedding = nn.Linear(self.input_dim, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_feedforward,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Output layer: Now produce two sets of outputs
        self.fc_out = nn.Linear(dim_feedforward, self.output_dim)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, embedding_dim)
        
        # Output shape: (batch_size, sequence_length, output_dim)
        x = self.fc_out(x)

        return x

def build_model(input_shape, output_shape, num_heads=8, num_layers=6, dim_feedforward=512, dropout=0.1, window=1):
    model = TransformerModel(input_shape, output_shape, num_heads, num_layers, dim_feedforward, dropout)
    return model
