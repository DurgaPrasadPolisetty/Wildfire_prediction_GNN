import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from stgcn_model import STGCN
from data_loader import load_graph

# ---------------- CONFIG ----------------
WINDOW = 3
HIDDEN_DIM = 32
THRESHOLD = 0.4

DATA_PATH = os.path.join("data", "processed", "node_features.csv")
NODES_PATH = os.path.join("data", "processed", "nodes.csv")
OUTPUT_DIR = os.path.join("outputs", "binary_maps")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")

print("Loading datasets...")
df = pd.read_csv(DATA_PATH)
nodes_df = pd.read_csv(NODES_PATH)

all_dates = sorted(df["date"].unique())

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

# -------- Load Model --------
print("Loading trained model...")
dummy_day = grouped.get_group(all_dates[0])
in_channels = build_tensor(dummy_day).shape[1]

model = STGCN(
    in_channels=in_channels,
    hidden_channels=HIDDEN_DIM,
    window_size=WINDOW
).to(device)

model.load_state_dict(torch.load("best_stgcn.pt"))
model.eval()

cmap = mcolors.ListedColormap(["white", "red"])

print("Generating maps...\n")

for idx in range(WINDOW, len(all_dates)):

    target_date = all_dates[idx]
    seq_dates = all_dates[idx-WINDOW:idx]

    print(f"Processing {target_date}...")

    x_seq = []
    for d in seq_dates:
        x_seq.append(build_tensor(grouped.get_group(d)))

    x_seq = torch.stack(x_seq)

    with torch.no_grad():
        out = model(x_seq, edge_index)
        probs = torch.softmax(out, dim=1)
        fire_prob = probs[:, 1].cpu().numpy()

    df_target = grouped.get_group(target_date).sort_values("node_id")
    df_target["fire_probability"] = fire_prob

    df_merged = df_target.merge(nodes_df, on="node_id")

    binary_fire = (df_merged["fire_probability"] > THRESHOLD).astype(int)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df_merged["lon"],
        df_merged["lat"],
        c=binary_fire,
        cmap=cmap,
        s=4
    )

    plt.title(f"Binary Wildfire Prediction (24h) - {target_date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    save_path = os.path.join(OUTPUT_DIR, f"binary_{target_date}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

print("\nAll maps generated successfully.")