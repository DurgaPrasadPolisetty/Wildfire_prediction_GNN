from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import requests
from scipy.spatial import KDTree

from stgcn_model import STGCN
from data_loader import load_graph

# ---------------- CONFIG ----------------

MODEL_PATH = "best_stgcn.pt"
NODE_PATH = "data/processed/nodes.csv"
FEATURE_PATH = "data/processed/node_features_full_year.csv"

WINDOW = 3
HIDDEN_DIM = 32

device = torch.device("cpu")

# ---------------- FLASK APP ----------------

app = Flask(__name__)
CORS(app)

# ---------------- LOAD NODE COORDINATES ----------------

print("Loading nodes...")

nodes = pd.read_csv(NODE_PATH)

node_coords = nodes[["lat", "lon"]].values

tree = KDTree(node_coords)

print("Nodes loaded:", len(nodes))

# ---------------- LOAD GRAPH ----------------

print("Loading graph...")

edge_index = load_graph()

# ---------------- LOAD MODEL ----------------

print("Loading trained model...")

df = pd.read_csv(FEATURE_PATH)

num_features = len(
    df.drop(columns=["node_id", "date", "fire", "ignition"], errors="ignore").columns
)

model = STGCN(
    in_channels=num_features,
    hidden_channels=HIDDEN_DIM,
    window_size=WINDOW
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully")

# ---------------- WEATHER API ----------------

def fetch_weather(lat, lon):

    try:

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

        r = requests.get(url)
        data = r.json()

        weather = data["current_weather"]

        temperature = weather.get("temperature", 0)
        wind_speed = weather.get("windspeed", 0)
        wind_dir = weather.get("winddirection", 0)

        # Convert wind direction → vector
        wind_u = wind_speed * np.cos(np.radians(wind_dir))
        wind_v = wind_speed * np.sin(np.radians(wind_dir))

        precipitation = 0
        humidity = 50

        return temperature, wind_u, wind_v, precipitation, wind_speed, humidity

    except:

        return 30, 0, 0, 0, 0, 50


# ---------------- PREPARE FEATURE WINDOW ----------------

print("Preparing node features...")

grouped = df.groupby("date")

dates = sorted(df["date"].unique())

latest_dates = dates[-WINDOW:]

def build_tensor(df_day):

    df_day = df_day.sort_values("node_id")

    feature_df = df_day.drop(
        columns=["node_id", "date", "fire", "ignition"],
        errors="ignore"
    )

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-6)

    x = torch.tensor(feature_df.values, dtype=torch.float)

    return x


daily_data = {
    d: build_tensor(grouped.get_group(d))
    for d in latest_dates
}

# ---------------- PREDICTION API ----------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    lat = float(data["lat"])
    lon = float(data["lon"])

    print("Prediction request:", lat, lon)

    # ---------------- NEAREST NODE ----------------

    dist, node_idx = tree.query([lat, lon])

    print("Nearest node:", node_idx)

    # ---------------- FETCH WEATHER ----------------

    temperature, wind_u, wind_v, precipitation, wind_speed, humidity = fetch_weather(lat, lon)

    # ---------------- BUILD TEMPORAL WINDOW ----------------

    x_seq = torch.stack([daily_data[d] for d in latest_dates])

    # ---------------- UPDATE NODE FEATURES ----------------

    if x_seq.shape[2] >= 4:

        x_seq[:, node_idx, 0] = temperature
        x_seq[:, node_idx, 1] = wind_u
        x_seq[:, node_idx, 2] = wind_v
        x_seq[:, node_idx, 3] = precipitation

    # ---------------- RUN MODEL ----------------

    with torch.no_grad():

        out = model(x_seq, edge_index)

        probs = torch.softmax(out, dim=1)

        fire_prob = float(probs[node_idx, 1])

    # ---------------- RISK LEVEL ----------------

    if fire_prob > 0.6:
        risk = "High"
    elif fire_prob > 0.3:
        risk = "Moderate"
    else:
        risk = "Low"

    # ---------------- RESPONSE ----------------

    return jsonify({

        "latitude": lat,
        "longitude": lon,
        "nearest_node": int(node_idx),

        "temperature": float(temperature),
        "wind_speed": float(wind_speed),
        "humidity": float(humidity),

        "fire_probability": fire_prob,
        "risk": risk
    })


# ---------------- ROOT ----------------

@app.route("/")
def home():

    return jsonify({
        "status": "Wildfire AI Server Running"
    })


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )