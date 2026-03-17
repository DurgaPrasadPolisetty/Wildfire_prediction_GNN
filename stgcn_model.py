import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Attention Network
    """

    def __init__(self, in_channels, hidden_channels, window_size):
        super(STGCN, self).__init__()

        self.window_size = window_size

        # Graph Attention Layers
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=4,
            concat=False
        )

        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=4,
            concat=False
        )

        # Temporal Model
        self.lstm = nn.LSTM(
            hidden_channels,
            hidden_channels,
            batch_first=True
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x_seq, edge_index, edge_weight=None):

        spatial_embeddings = []

        for t in range(self.window_size):

            x = x_seq[t]

            # Spatial Attention
            x = self.gat1(x, edge_index)
            x = torch.relu(x)

            x = self.gat2(x, edge_index)
            x = torch.relu(x)

            spatial_embeddings.append(x)

        # stack across time
        spatial_stack = torch.stack(spatial_embeddings, dim=1)

        # temporal learning
        lstm_out, _ = self.lstm(spatial_stack)

        final_emb = lstm_out[:, -1, :]

        out = self.classifier(final_emb)

        return out