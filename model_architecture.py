import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(MLPModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        
        for hidden_dim in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = torch.sigmoid(self.output_layer(x))
        return x