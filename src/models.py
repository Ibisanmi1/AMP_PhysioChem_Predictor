

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysioChemMLP(nn.Module):
    
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], 
                 dropout_rate: float = 0.3, output_dim: int = 1, use_residual: bool = True):
        
        super(PhysioChemMLP, self).__init__()
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0] if hidden_dims else input_dim),
            nn.LayerNorm(hidden_dims[0] if hidden_dims else input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        prev_dim = hidden_dims[0] if hidden_dims else input_dim
        
        for hidden_dim in hidden_dims[1:] if len(hidden_dims) > 1 else []:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.layers:
            if self.use_residual and x.shape[-1] == layer[0].out_features:
                x = x + layer(x)
            else:
                x = layer(x)
        
        return self.output_layer(x)


class PhysioChemCNN(nn.Module):
    
    
    def __init__(self, input_channels: int = 20, conv_channels: list = [64, 128, 256],
                 kernel_sizes: list = [3, 5, 7], dropout_rate: float = 0.3,
                 output_dim: int = 1, use_residual: bool = True):
        
        super(PhysioChemCNN, self).__init__()
        
        self.input_channels = input_channels
        self.use_residual = use_residual
        
        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_block = nn.Sequential(
                nn.Conv1d(prev_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.conv_layers.append(conv_block)
            prev_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(prev_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        for i, conv in enumerate(self.conv_layers):
            out = conv(x)
            if self.use_residual and x.shape[1] == out.shape[1]:
                x = x + out
            else:
                x = out
        
        x = self.global_pool(x).squeeze(-1)
        
        return self.fc(x)


class PhysioChemHybridModel(nn.Module):
    
    
    def __init__(self, seq_input_dim: int = 20, physchem_input_dim: int = 50,
                 seq_conv_channels: list = [64, 128, 256],
                 physchem_hidden_dims: list = [256, 128, 64],
                 dropout_rate: float = 0.3, output_dim: int = 1):
        
        super(PhysioChemHybridModel, self).__init__()
        
        self.seq_cnn = PhysioChemCNN(
            input_channels=seq_input_dim,
            conv_channels=seq_conv_channels,
            dropout_rate=dropout_rate,
            output_dim=64
        )
        
        self.physchem_mlp = PhysioChemMLP(
            input_dim=physchem_input_dim,
            hidden_dims=physchem_hidden_dims,
            dropout_rate=dropout_rate,
            output_dim=64
        )
        
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(2)
        ])
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, seq_features, physchem_features):
        
        seq_out = self.seq_cnn(seq_features)
        
        physchem_out = self.physchem_mlp(physchem_features)
        
        combined = torch.cat([seq_out, physchem_out], dim=1)
        
        fused = combined
        for fusion_layer in self.feature_fusion:
            fused_output = fusion_layer(combined)
            fused = fused + fused_output
        
        return self.combined_mlp(fused)


class PhysioChemLSTM(nn.Module):
    
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128,
                 num_layers: int = 2, dropout_rate: float = 0.3,
                 output_dim: int = 1, bidirectional: bool = True):
        
        super(PhysioChemLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        x = self.input_proj(x)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.fc(hidden)


class PhysioChemRNN(nn.Module):
    
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128,
                 num_layers: int = 2, dropout_rate: float = 0.3,
                 output_dim: int = 1):
        
        super(PhysioChemRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        x = self.input_proj(x)
        
        rnn_out, hidden = self.rnn(x)
        
        hidden = hidden[-1]
        
        return self.fc(hidden)


class PhysioChemGRU(nn.Module):
    
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128,
                 num_layers: int = 2, dropout_rate: float = 0.3,
                 output_dim: int = 1, bidirectional: bool = True):
        
        super(PhysioChemGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        x = self.input_proj(x)
        
        gru_out, hidden = self.gru(x)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.fc(hidden)


class CNNBiLSTMHybrid(nn.Module):
    
    
    def __init__(self, seq_input_dim: int = 20, 
                 seq_conv_channels: list = [64, 128, 256],
                 lstm_hidden_dim: int = 128, lstm_num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1):
        
        super(CNNBiLSTMHybrid, self).__init__()
        
        self.cnn = PhysioChemCNN(
            input_channels=seq_input_dim,
            conv_channels=seq_conv_channels,
            dropout_rate=dropout_rate,
            output_dim=64
        )
        
        self.bilstm = PhysioChemLSTM(
            input_dim=seq_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout_rate=dropout_rate,
            output_dim=64,
            bidirectional=True
        )
        
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(2)
        ])
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, seq_features):
        
        cnn_out = self.cnn(seq_features)
        
        lstm_input = seq_features.transpose(1, 2)
        lstm_out = self.bilstm(lstm_input)
        
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        
        fused = combined
        for fusion_layer in self.feature_fusion:
            fused_output = fusion_layer(combined)
            fused = fused + fused_output
        
        return self.combined_mlp(fused)


class PhysioChemTransformer(nn.Module):
    
    
    def __init__(self, input_dim: int = 20, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout_rate: float = 0.1,
                 output_dim: int = 1, max_length: int = 100):
        
        super(PhysioChemTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(1, max_length, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='gelu',  # GELU activation
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.virtual_node = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.virtual_node, std=0.02)
        
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            ) for _ in range(3)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        
        batch_size, seq_len, _ = x.shape
        
        x = self.input_proj(x)
        
        if seq_len <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :seq_len, :]
        
        virtual = self.virtual_node.expand(batch_size, 1, -1)
        x = torch.cat([virtual, x], dim=1)
        
        if mask is not None:
            virtual_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([virtual_mask, mask], dim=1)
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=(~mask.bool()))
        else:
            x = self.transformer(x)
        
        virtual_features = x[:, 0, :]
        
        fused_features = virtual_features
        for fusion_layer in self.feature_fusion:
            fused_features = fused_features + fusion_layer(virtual_features)
        
        return self.head(fused_features)


def get_model(model_type: str, **kwargs):
    
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return PhysioChemCNN(**kwargs)
    elif model_type == 'rnn':
        return PhysioChemRNN(**kwargs)
    elif model_type == 'bilstm' or model_type == 'lstm':
        if 'bidirectional' not in kwargs:
            kwargs['bidirectional'] = True
        return PhysioChemLSTM(**kwargs)
    elif model_type == 'bigru' or model_type == 'gru':
        return PhysioChemGRU(**kwargs)
    elif model_type == 'cnn_bilstm' or model_type == 'cnn_bilstm_hybrid':
        return CNNBiLSTMHybrid(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: cnn, rnn, bilstm, bigru, cnn_bilstm")


if __name__ == "__main__":
    batch_size = 4
    seq_length = 20
    
    print("Testing MLP...")
    mlp = PhysioChemMLP(input_dim=50, output_dim=1)
    x_mlp = torch.randn(batch_size, 50)
    out_mlp = mlp(x_mlp)
    print(f"MLP output shape: {out_mlp.shape}")
    
    print("\nTesting CNN...")
    cnn = PhysioChemCNN(input_channels=20, output_dim=1)
    x_cnn = torch.randn(batch_size, 20, seq_length)
    out_cnn = cnn(x_cnn)
    print(f"CNN output shape: {out_cnn.shape}")
    
    print("\nTesting Hybrid...")
    hybrid = PhysioChemHybridModel(seq_input_dim=20, physchem_input_dim=50, output_dim=1)
    x_seq = torch.randn(batch_size, 20, seq_length)
    x_physchem = torch.randn(batch_size, 50)
    out_hybrid = hybrid(x_seq, x_physchem)
    print(f"Hybrid output shape: {out_hybrid.shape}")
    
    print("\nTesting LSTM...")
    lstm = PhysioChemLSTM(input_dim=20, output_dim=1)
    x_lstm = torch.randn(batch_size, seq_length, 20)
    out_lstm = lstm(x_lstm)
    print(f"LSTM output shape: {out_lstm.shape}")
    
    print("\nTesting Transformer...")
    transformer = PhysioChemTransformer(input_dim=20, output_dim=1)
    x_transformer = torch.randn(batch_size, seq_length, 20)
    out_transformer = transformer(x_transformer)
    print(f"Transformer output shape: {out_transformer.shape}")
    
    print("\nAll models tested successfully!")

