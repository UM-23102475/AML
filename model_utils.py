import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(MLPModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        
        for hidden_dim in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            x = layer(x)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = torch.sigmoid(self.output_layer(x))
        
        return x