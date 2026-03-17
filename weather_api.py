import requests

API_KEY = "3882e2fa56da57ce9576a46e10993dd0"

def get_weather(lat, lon):

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    response = requests.get(url).json()

    weather = {
        "temperature": response["main"]["temp"],
        "humidity": response["main"]["humidity"],
        "wind_speed": response["wind"]["speed"],
        "precipitation": response.get("rain", {}).get("1h", 0)
    }

    return weather