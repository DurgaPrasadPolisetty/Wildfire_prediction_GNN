# 🔥 Wildfire Prediction GNN

A state-of-the-art Spatio-Temporal Graph Convolutional Network (STGCN) for predicting wildfire propagation. This project combines deep learning on graph structures with a modern interactive dashboard to visualize fire risks and spread patterns.

##  Overview

Wildfire propagation is a complex phenomenon influenced by spatial connectivity and temporal dynamics. This project leverages **Graph Neural Networks (GNNs)** to model the relationships between different geographical regions, treating them as nodes in a graph. By processing historical fire data, weather patterns, and terrain features, the STGCN model provides accurate forecasts of fire spread.

##  Features

- **STGCN Architecture**: Utilizes Spatio-Temporal Graph Convolutional Networks to capture both spatial dependencies and temporal trends.
- **Interactive Dashboard**: A React-based frontend providing real-time visualization of wildfire risks.
- **Data-Driven Insights**: Processes complex datasets including weather conditions and historical fire incidents.
- **Scalable Backend**: Flask/Python server for handling model inference and data processing.

##  Tech Stack

- **Backend**: Python, PyTorch, PyTorch Geometric, Flask.
- **Frontend**: React, Vite, CSS3.
- **Graph Processing**: NetworkX, Pandas, NumPy.
- **Visualization**: Matplotlib, Interactive Maps.

##  Project Structure

```text
wildfire_GNN/
├── src/                    # Backend source code
│   ├── train_stgcn.py      # STGCN Model training script
│   ├── server.py           # Backend Flask server
│   ├── model.py            # Model definitions
│   └── data_loader.py      # Data preprocessing utilities
├── wildfire-dashboard/     # React Frontend application
├── data/                   # Dataset storage
├── outputs/                # Model checkpoints and results
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

##  Setup & Installation

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/DurgaPrasadPolisetty/Wildfire_prediction_GNN.git
   cd Wildfire_prediction_GNN
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python src/server.py
   ```

### Frontend Setup
1. Navigate to the dashboard directory:
   ```bash
   cd wildfire-dashboard
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

##  Model Training
To train the STGCN model on your dataset, run:
```bash
python src/train_stgcn.py
```

##  License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed by [Durga Prasad Polisetty](https://github.com/DurgaPrasadPolisetty)*
