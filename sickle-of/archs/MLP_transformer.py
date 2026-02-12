import torch
import torch.nn as nn
import torch.optim as optim

class MLP_transformer(nn.Module):
    def __init__(self, input_shape, output_shape, units1=2000, units2=8000, units3=2000, activation='elu', num_heads=8, num_layers=6, dim_feedforward=16, dropout=0.1):
        super(MLP_transformer, self).__init__()
        """
        input: [B, T, C, Samples]
        output: [B, T', C', H, W, D]
        """
        print(f"Input shape: {input_shape}; output shape: {output_shape}")
        self.output_shape = output_shape
        self.input_shape = input_shape

        # Encoder-embedding (MLP)
        samples = int(input_shape[-1])
        self.fc1 = nn.Linear(samples, 1)
        # self.fc2 = nn.Linear(units1, units2)
        # self.fc3 = nn.Linear(units2, units3)
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError("Unsupported activation function")
        
        # Transformer
        self.embedding = nn.Linear(self.input_shape[-2], dim_feedforward)
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

        # Decoder (CNN)
        in_channel = self.input_shape[-3] * dim_feedforward  # T*embd
        out_channel = self.output_shape[-5] * self.output_shape[-4]  # T'*C'
        self.conv_transpose = nn.Sequential(
                                            nn.ConvTranspose3d(
                                                in_channels=in_channel,
                                                out_channels=out_channel,
                                                kernel_size=(output_shape[-3], output_shape[-2], output_shape[-1]),  # Controls the output size
                                                stride=(1, 1, 1),  # Controls upsampling
                                                padding=(0, 0, 0),  # No padding
                                                output_padding=0  # Adjust as needed
                                            ), 
                                            self.activation
                                            )

    def forward(self, x):
        # Input encoder: [B,T,C,Samples] -> [B,T,C]
        x = self.activation(self.fc1(x)) # [B,T,C,Samp] -> [B,T,C,1]
        x = torch.squeeze(x, dim=-1) # [B,T,C]

        # Tranformer: [B,T,C] -> [B,T,emb]
        x = self.embedding(x) # [B,T,C] -> [B,T,emb]
        x = self.transformer_encoder(x) # [B,T,emb]
        
        # Output decoder: [B,T,emb] -> [B,T,C,H,W,D]
        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [B,T,emb] -> [B,T,emb,1,1,1]
        # Combine T and C dimensions
        batch_size, time, _, _,_,_ = x.shape
        in_channel = self.conv_transpose[0].in_channels
        x = x.reshape(batch_size, in_channel, 1, 1, 1) # [B, T*emb, 1, 1, 1]
        # loop over ConvTranspose
        x = self.conv_transpose(x) # [B, T*emb, 1, 1, 1] -> [B, T'*C', H, W, D]
        # Reshape to [B, T', C', H, W, D]
        out_H, out_W, out_D = self.conv_transpose[0].kernel_size
        channels = self.output_shape[-4]
        time = self.output_shape[-5]
        x = x.reshape(batch_size, time, channels, out_H, out_W, out_D)

        return x

def build_model(input_shape, output_shape, units1=2000, units2=8000, units3=2000, activation='elu', window=1, num_heads=8, num_layers=6, dim_feedforward=16, dropout=0.1):
    model = MLP_transformer(input_shape, output_shape, units1=units1, units2=units2, units3=units3, activation=activation, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
    return model

def get_meta_model(input_shape, output_shape):
    def meta_model(hp):
        units1 = hp.Int('units1', min_value=2, max_value=8000, step=800)
        units2 = hp.Int('units2', min_value=2, max_value=8000, step=800)
        units3 = hp.Int('units3', min_value=2, max_value=8000, step=800)
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        window = hp.Int('window', min_value=1, max_value=100, step=10)
        num_heads = hp.Int('num_heads', min_value=1, max_value=100, step=10)#8
        num_layers = hp.Int('num_layers', min_value=1, max_value=100, step=10)#6
        dim_feedforward = hp.Int('dim_feedforward', min_value=1, max_value=100, step=10)#16
        dropout = hp.float('dropout', min_value=0.01, max_value=1, step=0.02)#0.1
        return build_model(input_shape, output_shape, units1=units1, units2=units2, units3=units3, activation=activation, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
    return meta_model
