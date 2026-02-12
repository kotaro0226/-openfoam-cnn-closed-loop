import torch
import torch.nn as nn

class CNN_transformer(nn.Module):
    def __init__(self, input_shape, output_shape, num_heads=8, num_layers=6, dim_feedforward=16, kernel_size=3, dropout=0.1, activation='elu'):
        """
        input: [B, T, C, H, W, D]
        output: [B, T', C', H, W, D]
        """
        super(CNN_transformer, self).__init__()
        print(f"Input shape: {input_shape}; Output shape: {output_shape}")

        self.input_shape = input_shape
        self.output_shape = output_shape
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError("Unsupported activation function")

        # CNN feature extraction
        self.conv1 = nn.Conv3d(in_channels=self.input_shape[-4], out_channels=dim_feedforward, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv3d(in_channels=dim_feedforward, out_channels=dim_feedforward, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # Transformer
        self.embedding = nn.Linear(dim_feedforward*self.input_shape[-1]*self.input_shape[-2]*self.input_shape[-3], dim_feedforward)  # Project CNN features to transformer embedding size
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_feedforward,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=self.activation,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.unembedding = nn.Linear(dim_feedforward, dim_feedforward*self.input_shape[-1]*self.input_shape[-2]*self.input_shape[-3])  # Project back from transformer embedding size

        # CNN feature reconstruction
        self.deconv1 = nn.Conv3d(in_channels=dim_feedforward*self.input_shape[-5], out_channels=dim_feedforward, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        out_channel = self.output_shape[-5] * self.output_shape[-4]  # T'*C'
        self.deconv2 = nn.Conv3d(in_channels=dim_feedforward, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # Input shape: [B, T, C, H, W, D]

        # Step 1: CNN feature extraction
        batch_size, time_steps, _, height, width, depth = x.shape
        x = x.reshape(batch_size * time_steps, self.input_shape[-4], height, width, depth)  # Combine batch and time
        x = self.activation(self.conv1(x))  # [B*T, C, H, W, D] -> [B*T, dim_feedforward, H, W, D]
        x = self.activation(self.conv2(x))  # [B*T, dim_feedforward, H, W, D] -> [B*T, dim_feedforward, H, W, D]
        x = x.reshape(batch_size, time_steps, -1, height, width, depth)  # Restore batch and time: [B, T, dim_feedforward, H, W, D]

        # Step 2: Transformer for temporal and channel relationships
        x_flat = x.view(batch_size, time_steps, -1)  # Flatten spatial dimensions for transformer [B, T, dim_feedforward*H*W*D]
        x_emb = self.embedding(x_flat)  # Project to transformer embedding space: [B, T, dim_feedforward*H*W*D] -> [B, T, dim_feedforward]
        x_trans = self.transformer_encoder(x_emb)  # Apply transformer [B, T, dim_feedforward]
        x_trans = self.unembedding(x_trans)  # Project back to [B, T, dim_feedforward] -> [B, T, dim_feedforward*H*W*D]
        x = x_trans.view(batch_size, time_steps, -1, height, width, depth)  # Reshape to match spatial dimensions: [B, T, dim_feedforward, H, W, D]

        # Step 3: CNN feature reconstruction
        emb = x.shape[-4]
        x = x.reshape(batch_size, time_steps*emb, height, width, depth)  # Combine time and dim_feedforward
        x = self.activation(self.deconv1(x))  # [B, T*dim_feedforward, H, W, D] -> [B, dim_feedforward, H, W, D]
        x = self.activation(self.deconv2(x))  # [B, dim_feedforward, H, W, D] -> [B, T'*C', H, W, D]
        # Reshape to [B, T', C', H, W, D]
        channels = self.output_shape[-4]
        time = self.output_shape[-5]
        x = x.reshape(batch_size, time, channels, height, width, depth)  # Restore batch and time

        # Output shape: [B, T', C', H, W, D]
        return x

def build_model(input_shape, output_shape, kernel_size=3, activation='elu', window=1, num_heads=2, num_layers=2, dim_feedforward=4, dropout=0.1):
    model = CNN_transformer(input_shape, output_shape, kernel_size=kernel_size, activation=activation, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
    return model

def get_meta_model(input_shape, output_shape):
    def meta_model(hp):
        kernel_size = hp.Int('kernel_size', min_value=2, max_value=10, step=1)
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        window = hp.Int('window', min_value=1, max_value=100, step=10)
        num_heads = hp.Int('num_heads', min_value=1, max_value=100, step=10)#8
        num_layers = hp.Int('num_layers', min_value=1, max_value=100, step=10)#6
        dim_feedforward = hp.Int('dim_feedforward', min_value=1, max_value=100, step=10)#16
        dropout = hp.float('dropout', min_value=0.01, max_value=1, step=0.02)#0.1
        return build_model(input_shape, output_shape, kernel_size=kernel_size, activation=activation, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
    return meta_model


