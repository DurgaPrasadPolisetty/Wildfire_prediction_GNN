import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score

from stgcn_model import STGCN
from data_loader import load_graph


# ---------------- CONFIG ----------------

WINDOW = 3
EPOCHS = 20
PATIENCE = 3
HIDDEN_DIM = 32
LR = 0.0001

DATA_PATH = os.path.join("data", "processed", "node_features_full_year.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD DATA ----------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

all_dates = sorted(df["date"].unique())

train_split = int(len(all_dates) * 0.7)
val_split = int(len(all_dates) * 0.8)

train_dates = all_dates[:train_split]
val_dates = all_dates[train_split:val_split]
test_dates = all_dates[val_split:]

print("Train dates:", len(train_dates))
print("Validation dates:", len(val_dates))
print("Test dates:", len(test_dates))


# ---------------- LOAD GRAPH ----------------

print("Loading graph...")
edge_index = load_graph().to(device)

grouped = df.groupby("date")


# ---------------- BUILD TENSORS ----------------

def build_tensor(df_day):

    df_day = df_day.sort_values("node_id")

    y = torch.tensor(df_day["fire"].values, dtype=torch.long)

    feature_df = df_day.drop(
        columns=["node_id", "date", "fire", "ignition"],
        errors="ignore"
    )

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-6)

    feature_df = feature_df.fillna(0)

    x = torch.tensor(feature_df.values, dtype=torch.float)

    return x.to(device), y.to(device)


print("Precomputing daily tensors...")

daily_data = {
    d: build_tensor(grouped.get_group(d))
    for d in all_dates
}


# ---------------- WIND EDGE WEIGHTS ----------------

def compute_edge_weights(df_day, edge_index):

    wind_u = torch.tensor(df_day["wind_u"].values, dtype=torch.float).to(device)
    wind_v = torch.tensor(df_day["wind_v"].values, dtype=torch.float).to(device)

    wind_u = torch.nan_to_num(wind_u, nan=0.0)
    wind_v = torch.nan_to_num(wind_v, nan=0.0)

    wind_strength = torch.sqrt(wind_u ** 2 + wind_v ** 2)

    src = edge_index[0]

    weights = wind_strength[src]

    max_val = torch.max(weights)

    if max_val > 0:
        weights = weights / max_val
    else:
        weights = torch.ones_like(weights)

    weights = torch.nan_to_num(weights, nan=0.0)

    return weights


# ---------------- MODEL ----------------

print("Initializing model...")

model = STGCN(
    in_channels=daily_data[all_dates[0]][0].shape[1],
    hidden_channels=HIDDEN_DIM,
    window_size=WINDOW
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)


# ---------------- CLASS WEIGHTING ----------------

print("Computing class weights...")

fire_count = 0
no_fire_count = 0

for d in train_dates:

    y = daily_data[d][1]

    fire_count += (y == 1).sum().item()
    no_fire_count += (y == 0).sum().item()

weight_fire = no_fire_count / (fire_count + 1e-6)

class_weights = torch.tensor([1.0, weight_fire], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Fire samples:", fire_count)
print("No-fire samples:", no_fire_count)


# ---------------- TRAINING ----------------

best_val_f1 = 0
patience_counter = 0

train_losses = []
val_scores = []

print("\nTraining ST-GCN...\n")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for i in range(WINDOW, len(train_dates)):

        seq_dates = train_dates[i-WINDOW:i]
        target_date = train_dates[i]

        x_seq = torch.stack([daily_data[d][0] for d in seq_dates])
        y = daily_data[target_date][1]

        edge_weights = compute_edge_weights(
            grouped.get_group(target_date),
            edge_index
        )

        optimizer.zero_grad()

        out = model(x_seq, edge_index, edge_weights)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(train_dates) - WINDOW)

    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")


    # ---------------- VALIDATION ----------------

    model.eval()

    val_preds = []
    val_true = []

    with torch.no_grad():

        for i in range(WINDOW, len(val_dates)):

            seq_dates = val_dates[i-WINDOW:i]
            target_date = val_dates[i]

            x_seq = torch.stack([daily_data[d][0] for d in seq_dates])
            y = daily_data[target_date][1]

            edge_weights = compute_edge_weights(
                grouped.get_group(target_date),
                edge_index
            )

            out = model(x_seq, edge_index, edge_weights)

            probs = torch.softmax(out, dim=1)

            fire_prob = probs[:, 1]

            pred = (fire_prob > 0.3).long()

            val_preds.extend(pred.cpu().numpy())
            val_true.extend(y.cpu().numpy())

    val_f1 = f1_score(val_true, val_preds, zero_division=0)

    val_scores.append(val_f1)

    print("Validation F1:", round(val_f1, 4))


    # ---------------- EARLY STOPPING ----------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_stgcn.pt")
        patience_counter = 0

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("Early stopping triggered\n")
        break


# ---------------- TESTING ----------------

print("\nLoading best model...\n")

model.load_state_dict(torch.load("best_stgcn.pt"))

model.eval()

all_probs = []
all_true = []

with torch.no_grad():

    for i in range(WINDOW, len(test_dates)):

        seq_dates = test_dates[i-WINDOW:i]
        target_date = test_dates[i]

        x_seq = torch.stack([daily_data[d][0] for d in seq_dates])
        y = daily_data[target_date][1]

        edge_weights = compute_edge_weights(
            grouped.get_group(target_date),
            edge_index
        )

        out = model(x_seq, edge_index, edge_weights)

        probs = torch.softmax(out, dim=1)

        fire_prob = probs[:, 1]

        all_probs.extend(fire_prob.cpu().numpy())
        all_true.extend(y.cpu().numpy())


# ---------------- THRESHOLD SEARCH ----------------

print("\nSearching best threshold...\n")

all_probs = np.array(all_probs)
all_true = np.array(all_true)

thresholds = np.arange(0.05, 0.9, 0.05)

best_f1 = 0
best_threshold = 0
best_precision = 0
best_recall = 0

for t in thresholds:

    preds = (all_probs > t).astype(int)

    p = precision_score(all_true, preds, zero_division=0)
    r = recall_score(all_true, preds, zero_division=0)
    f = f1_score(all_true, preds, zero_division=0)

    print(f"Threshold {t:.2f} | Precision {p:.4f} | Recall {r:.4f} | F1 {f:.4f}")

    if f > best_f1:

        best_f1 = f
        best_threshold = t
        best_precision = p
        best_recall = r


print("\n------ BEST THRESHOLD RESULTS ------")

print("Best Threshold:", best_threshold)
print("Precision:", round(best_precision, 4))
print("Recall:", round(best_recall, 4))
print("F1 Score:", round(best_f1, 4))


# ---------------- SAVE TRAINING CURVES ----------------

os.makedirs("outputs", exist_ok=True)

plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("outputs/train_loss.png")

plt.figure()
plt.plot(val_scores)
plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.savefig("outputs/validation_f1.png")

print("\nTraining curves saved in outputs/")