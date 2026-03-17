export default function PredictionCard({ prediction }) {

  if (!prediction)
    return (
      <div className="prediction-card">
        <h2>🔥 Wildfire Prediction</h2>
        <p>Click anywhere on the map to predict wildfire risk.</p>
      </div>
    );

  const riskColor =
    prediction.risk === "High"
      ? "red"
      : prediction.risk === "Moderate"
      ? "orange"
      : "green";

  return (

    <div className="prediction-card">

      <h2>🔥 Wildfire Prediction</h2>

      <p><b>Latitude:</b> {prediction.latitude}</p>
      <p><b>Longitude:</b> {prediction.longitude}</p>

      <hr />

      <h3>Weather</h3>

      <p><b>Temperature:</b> {prediction.temperature} °C</p>
      <p><b>Wind Speed:</b> {prediction.wind_speed} m/s</p>
      <p><b>Humidity:</b> {prediction.humidity}%</p>

      <hr />

      <h3>AI Prediction</h3>

      <p><b>Fire Probability:</b> {prediction.fire_probability.toFixed(3)}</p>

      <h2 style={{ color: riskColor }}>
        {prediction.risk} Risk
      </h2>

    </div>
  );
}