import requests
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append("src")

from stgcn_model import STGCN

# ---------------- CONFIG ----------------
API_KEY = "3882e2fa56da57ce9576a46e10993dd0"
CITY = "Chennai"
WINDOW = 3
HIDDEN_DIM = 32
THRESHOLD = 0.4

device = torch.device("cpu")

# ---------------- STEP 1: Fetch Weather ----------------
print("Fetching live weather...")

url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
response = requests.get(url)
data = response.json()

temperature = data["main"]["temp"]
wind_speed = data["wind"]["speed"]
wind_deg = data["wind"]["deg"]
precipitation = data.get("rain", {}).get("1h", 0)

# Convert wind speed + direction to u/v
import math
wind_u = wind_speed * math.cos(math.radians(wind_deg))
wind_v = wind_speed * math.sin(math.radians(wind_deg))

print("Weather:")
print("Temp:", temperature)
print("Wind U:", wind_u)
print("Wind V:", wind_v)
print("Rain:", precipitation)

# ---------------- STEP 2: Static Features ----------------
# Use example static features (you can replace with real node values)

ndvi = 0.45
elevation = 50
slope = 2.5

features = [
    temperature,
    wind_u,
    wind_v,
    precipitation,
    ndvi,
    elevation,
    slope
]

# ---------------- STEP 3: Prepare Tensor ----------------
x = torch.tensor(features, dtype=torch.float).unsqueeze(0)

# Fake temporal window (repeat same for demo)
x_seq = torch.stack([x for _ in range(WINDOW)])

# ---------------- STEP 4: Load Model ----------------
model = STGCN(
    in_channels=7,
    hidden_channels=HIDDEN_DIM,
    window_size=WINDOW
).to(device)

model.load_state_dict(torch.load("best_stgcn.pt", map_location=device))
model.eval()

# Dummy edge index (single node)
edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# ---------------- STEP 5: Predict ----------------
with torch.no_grad():
    out = model(x_seq, edge_index)
    prob = F.softmax(out, dim=1)[0][1].item()

print("\nFire Probability:", round(prob, 4))

if prob > THRESHOLD:
    print("🔥 WILL BURN in next 24 hours")
else:
    print("🟢 SAFE")