import { useState } from "react";
import MapView from "./components/MapView";
import PredictionCard from "./components/PredictionCard";
import { predictFire } from "./services/api";
import "./App.css";

function App() {

  const [prediction, setPrediction] = useState(null);
  const [city, setCity] = useState("");

  async function searchCity() {

    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&q=${city}`
    );

    const data = await res.json();

    if (data.length > 0) {

      const lat = parseFloat(data[0].lat);
      const lon = parseFloat(data[0].lon);

      const result = await predictFire(lat, lon);

      setPrediction(result);

    }

  }

  return (

    <div className="app">

      <header className="header">
        🔥 AI Wildfire Monitoring Dashboard
      </header>

      <div className="search-bar">

        <input
          placeholder="Enter city name"
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />

        <button onClick={searchCity}>
          Predict Fire Risk
        </button>

      </div>

      <div className="dashboard">

        <div className="map-section">
          <MapView setPrediction={setPrediction} />
        </div>

        <div className="panel-section">
          <PredictionCard prediction={prediction} />
        </div>

      </div>

    </div>
  );
}

export default App;