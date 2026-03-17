import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score

from model import WildfireGCN
from data_loader import load_graph

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD DATA ONCE ----------------
print("Loading full dataset...")

DATA_PATH = os.path.join("data", "processed", "node_features.csv")
df = pd.read_csv(DATA_PATH)

all_dates = sorted(df["date"].unique())

# 70% train, 10% val, 20% test
train_split = int(len(all_dates) * 0.7)
val_split = int(len(all_dates) * 0.8)

train_dates = all_dates[:train_split]
val_dates = all_dates[train_split:val_split]
test_dates = all_dates[val_split:]

print("Train dates:", len(train_dates))
print("Validation dates:", len(val_dates))
print("Test dates:", len(test_dates))

# ---------------- LOAD EDGE INDEX ONCE ----------------
edge_index = load_graph().to(device)

# ---------------- GROUP DATA BY DATE ----------------
print("\nGrouping dataset by date...")
grouped = df.groupby("date")

def build_graph_from_df(df_day):
    df_day = df_day.sort_values("node_id")

    # Labels
    y = torch.tensor(df_day["fire"].values, dtype=torch.long)

    # Features
    feature_df = df_day.drop(
        columns=["node_id", "date", "fire", "ignition"],
        errors="ignore"
    )

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-6)

    x = torch.tensor(feature_df.values, dtype=torch.float)

    return Data(x=x.to(device),
                edge_index=edge_index,
                y=y.to(device))

print("Building graph snapshots...")

train_graphs = [build_graph_from_df(grouped.get_group(d)) for d in train_dates]
val_graphs   = [build_graph_from_df(grouped.get_group(d)) for d in val_dates]
test_graphs  = [build_graph_from_df(grouped.get_group(d)) for d in test_dates]

print("Preloading complete.\n")

# ---------------- INITIALIZE MODEL ----------------
sample_data = train_graphs[0]

model = WildfireGCN(
    in_channels=sample_data.x.shape[1],
    hidden_channels=64
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---------------- CLASS WEIGHTS ----------------
num_fire = (sample_data.y == 1).sum().item()
num_no_fire = (sample_data.y == 0).sum().item()

weight_fire = num_no_fire / (num_fire + 1e-6)
class_weights = torch.tensor([1.0, weight_fire], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ---------------- EARLY STOPPING ----------------
best_val_f1 = 0
patience = 5
patience_counter = 0

print("Training started...\n")

for epoch in range(50):

    # ---------- TRAIN ----------
    model.train()
    total_loss = 0

    for data in train_graphs:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_graphs)

    # ---------- VALIDATION ----------
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in val_graphs:
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(data.y.cpu().numpy())

    val_precision = precision_score(all_true, all_preds, zero_division=0)
    val_recall = recall_score(all_true, all_preds, zero_division=0)
    val_f1 = f1_score(all_true, all_preds, zero_division=0)

    print(f"Epoch {epoch+1:02d} | "
          f"Loss: {avg_loss:.4f} | "
          f"Val F1: {val_f1:.4f} | "
          f"Val Recall: {val_recall:.4f}")

    # ---------- EARLY STOPPING ----------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

print("\nTraining completed.\n")

# ---------------- TESTING ----------------
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for data in test_graphs:
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_true.extend(data.y.cpu().numpy())

precision = precision_score(all_true, all_preds, zero_division=0)
recall = recall_score(all_true, all_preds, zero_division=0)
f1 = f1_score(all_true, all_preds, zero_division=0)

print("------ FINAL TEST RESULTS ------")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))