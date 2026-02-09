
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbeddingCNN(nn.Module):
    
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 conv_channels: list = [64, 128, 256], kernel_sizes: list = [3, 5, 7],
                 dropout_rate: float = 0.3, output_dim: int = 1):
        
        super(EmbeddingCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        self.conv_layers = nn.ModuleList()
        prev_channels = embedding_dim
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.conv_layers.append(conv)
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
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        x = self.embedding(x)
        
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.global_pool(x).squeeze(-1)
        
        return self.fc(x)


class EmbeddingLSTM(nn.Module):
    
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1,
                 bidirectional: bool = True):
        
        super(EmbeddingLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
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
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        x = self.embedding(x)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.fc(hidden)


class EmbeddingGRU(nn.Module):
    
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1,
                 bidirectional: bool = True):
        
        super(EmbeddingGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
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
        
        x = self.embedding(x)
        
        gru_out, hidden = self.gru(x)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.fc(hidden)


class EmbeddingRNN(nn.Module):
    
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1):
        
        super(EmbeddingRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers,
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
        
        x = self.embedding(x)
        
        rnn_out, hidden = self.rnn(x)
        
        hidden = hidden[-1]
        
        return self.fc(hidden)


class EmbeddingCNNBiLSTMHybrid(nn.Module):
    
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 conv_channels: list = [64, 128], kernel_sizes: list = [3, 5],
                 lstm_hidden_dim: int = 128, lstm_num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1):
        
        super(EmbeddingCNNBiLSTMHybrid, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        self.cnn_layers = nn.ModuleList()
        prev_channels = embedding_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.cnn_layers.append(conv)
            prev_channels = out_channels
        
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_fc = nn.Sequential(
            nn.Linear(prev_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.bilstm = nn.LSTM(
            embedding_dim, lstm_hidden_dim, lstm_num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
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
        
        for name, param in self.bilstm.named_parameters():
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
        
        x_emb = self.embedding(x)
        
        cnn_input = x_emb.transpose(1, 2)
        for conv in self.cnn_layers:
            cnn_input = conv(cnn_input)
        cnn_out = self.cnn_pool(cnn_input).squeeze(-1)
        cnn_out = self.cnn_fc(cnn_out)
        
        lstm_out, (hidden, cell) = self.bilstm(x_emb)
        lstm_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        lstm_out = self.lstm_fc(lstm_hidden)
        
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        
        fused = combined
        for fusion_layer in self.feature_fusion:
            fused_output = fusion_layer(combined)
            fused = fused + fused_output
        
        return self.combined_mlp(fused)


def get_embedding_model(model_type: str, vocab_size: int = 21, **kwargs):
    
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return EmbeddingCNN(vocab_size=vocab_size, **kwargs)
    elif model_type == 'rnn':
        return EmbeddingRNN(vocab_size=vocab_size, **kwargs)
    elif model_type == 'lstm' or model_type == 'bilstm':
        if 'bidirectional' not in kwargs:
            kwargs['bidirectional'] = True
        return EmbeddingLSTM(vocab_size=vocab_size, **kwargs)
    elif model_type == 'gru' or model_type == 'bigru':
        if 'bidirectional' not in kwargs:
            kwargs['bidirectional'] = True
        return EmbeddingGRU(vocab_size=vocab_size, **kwargs)
    elif model_type == 'cnn_bilstm':
        return EmbeddingCNNBiLSTMHybrid(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}. Supported: cnn, rnn, lstm, bilstm, gru, bigru, cnn_bilstm")

