import os
import requests
import logging

logger = logging.getLogger(__name__)

def get_lat_lon(city_name):
    """
    Fetches latitude and longitude for a given city in Alaska using Google Maps Geocoding API.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("GOOGLE_MAPS_API_KEY not set.")
        return None, None

    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": f"{city_name}, AK",
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            logger.error(f"Geocoding error for {city_name}: {data['status']}")
            return None, None
    except Exception as e:
        logger.error(f"Error fetching coordinates for {city_name}: {e}")
        return None, None

def get_weather_forecast(city_name):
    """
    Fetches the current temperature and short forecast for a city.
    """
    lat, lon = get_lat_lon(city_name)
    if not lat or not lon:
        return {"city": city_name, "temp": "--", "condition": "Location not found"}

    headers = {
        "User-Agent": "(myweatherapp.com, contact@myweatherapp.com)"
    }

    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        point_resp = requests.get(points_url, headers=headers)
        point_resp.raise_for_status()
        point_data = point_resp.json()
        
        forecast_url = point_data['properties']['forecast']
        forecast_resp = requests.get(forecast_url, headers=headers)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
        
        current_period = forecast_data['properties']['periods'][0]
        
        return {
            "city": city_name,
            "temp": f"{current_period['temperature']}°{current_period['temperatureUnit']}",
            "condition": current_period['shortForecast'],
            "icon": current_period.get('icon')
        }

    except Exception as e:
        logger.error(f"Error fetching weather for {city_name}: {e}")
        return {"city": city_name, "temp": "--", "condition": "Error fetching data"}

def get_alaska_weather():
    """
    Fetches weather for the top 5 Alaskan cities.
    """
    cities = ["Anchorage", "Fairbanks", "Juneau", "Wasilla", "Sitka"]
    results = []
    for city in cities:
        results.append(get_weather_forecast(city))
    return results
