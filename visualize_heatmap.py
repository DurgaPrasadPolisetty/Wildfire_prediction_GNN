import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stgcn_model import STGCN
from data_loader import load_graph

WINDOW = 3
HIDDEN_DIM = 32

DATA_PATH = os.path.join("data", "processed", "node_features.csv")
NODES_PATH = os.path.join("data", "processed", "nodes.csv")

device = torch.device("cpu")

# Load datasets
df = pd.read_csv(DATA_PATH)
nodes_df = pd.read_csv(NODES_PATH)

all_dates = sorted(df["date"].unique())
target_date = all_dates[-1]

edge_index = load_graph().to(device)
grouped = df.groupby("date")

def build_tensor(df_day):
    df_day = df_day.sort_values("node_id")

    feature_df = df_day.drop(
        columns=["node_id", "date", "fire", "ignition"],
        errors="ignore"
    )

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-6)

    x = torch.tensor(feature_df.values, dtype=torch.float)
    return x.to(device)

# Prepare temporal window
idx = all_dates.index(target_date)
seq_dates = all_dates[idx-WINDOW:idx]

x_seq = []
for d in seq_dates:
    x_seq.append(build_tensor(grouped.get_group(d)))

x_seq = torch.stack(x_seq)

# Load model
model = STGCN(
    in_channels=x_seq.shape[2],
    hidden_channels=HIDDEN_DIM,
    window_size=WINDOW
).to(device)

model.load_state_dict(torch.load("best_stgcn.pt"))
model.eval()

with torch.no_grad():
    out = model(x_seq, edge_index)
    probs = torch.softmax(out, dim=1)
    fire_prob = probs[:, 1].cpu().numpy()

# Merge predictions with coordinates
df_target = grouped.get_group(target_date).sort_values("node_id")
df_target["fire_probability"] = fire_prob

df_merged = df_target.merge(nodes_df, on="node_id")

# Plot real geographic heatmap
plt.figure(figsize=(10, 8))

sc = plt.scatter(
    df_merged["lon"],
    df_merged["lat"],
    c=df_merged["fire_probability"],
    cmap="hot",
    s=5
)

plt.colorbar(sc, label="Fire Probability")
plt.title(f"Wildfire Risk Map - {target_date}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()