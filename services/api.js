export async function predictFire(lat, lon) {

  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ lat, lon }),
  });

  return await res.json();
}