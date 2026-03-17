import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class WildfireGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(WildfireGCN, self).__init__()

        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)

        x = self.gcn2(x, edge_index)
        x = torch.relu(x)

        x = self.classifier(x)

        return x