import torch
import numpy as np

from stgcn_model import STGCN
from data_loader import load_graph


# ---------------- CONFIG ----------------

MODEL_PATH = "best_stgcn.pt"

WINDOW = 3
HIDDEN_DIM = 32
INPUT_FEATURES = 9
THRESHOLD = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODEL ----------------

print("Loading trained model...")

model = STGCN(
    in_channels=INPUT_FEATURES,
    hidden_channels=HIDDEN_DIM,
    window_size=WINDOW
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully")


# ---------------- LOAD GRAPH ----------------

print("Loading graph structure...")

edge_index = load_graph().to(device)


# ---------------- USER INPUT ----------------

def get_user_input():

    print("\nEnter environmental conditions:\n")

    temperature = float(input("Temperature: "))
    wind_u = float(input("Wind U: "))
    wind_v = float(input("Wind V: "))
    precipitation = float(input("Precipitation: "))
    ndvi = float(input("NDVI: "))
    elevation = float(input("Elevation: "))
    slope = float(input("Slope: "))
    humidity = float(input("Humidity: "))
    vegetation = float(input("Vegetation Index: "))

    return np.array([
        temperature,
        wind_u,
        wind_v,
        precipitation,
        ndvi,
        elevation,
        slope,
        humidity,
        vegetation
    ])


# ---------------- FEATURE MATRIX ----------------

def create_feature_matrix(features, num_nodes):

    # replicate input features for all nodes
    x = np.tile(features, (num_nodes, 1))

    return torch.tensor(x, dtype=torch.float).to(device)


# ---------------- EDGE WEIGHTS ----------------

def compute_edge_weights(wind_u, wind_v, edge_index):

    wind_strength = np.sqrt(wind_u**2 + wind_v**2)

    weights = torch.ones(edge_index.shape[1]).to(device)

    weights = weights * wind_strength

    weights = weights / (weights.max() + 1e-6)

    return weights


# ---------------- PREDICTION ----------------

def predict_fire(features):

    num_nodes = int(edge_index.max().item()) + 1

    x_seq = []

    for _ in range(WINDOW):

        x = create_feature_matrix(features, num_nodes)
        x_seq.append(x)

    x_seq = torch.stack(x_seq)

    wind_u = features[1]
    wind_v = features[2]

    edge_weights = compute_edge_weights(wind_u, wind_v, edge_index)

    with torch.no_grad():

        out = model(x_seq, edge_index, edge_weights)

        probs = torch.softmax(out, dim=1)

        fire_prob = probs[:, 1].cpu().numpy()

    return fire_prob


# ---------------- MAIN ----------------

if __name__ == "__main__":

    features = get_user_input()

    print("\nRunning wildfire prediction...")

    fire_probabilities = predict_fire(features)

    avg_risk = fire_probabilities.mean()

    print("\nAverage Fire Probability:", round(avg_risk, 4))

    if avg_risk > THRESHOLD:

        print("🔥 High wildfire risk detected")

    else:

        print("✅ Low wildfire risk")