import os
import requests
import logging

logger = logging.getLogger(__name__)


def get_lat_lon(city_name, state=None):
    """
    Fetches latitude and longitude for a given city using Google Maps Geocoding API.

    Args:
        city_name: Name of the city
        state: State abbreviation (e.g., "AK"). If None, will search broadly.

    Returns:
        Tuple of (lat, lon, is_in_alaska, full_location_name)
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("GOOGLE_MAPS_API_KEY not set.")
        return None, None, False, None

    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    # Try with state first if provided, otherwise just city name
    address = f"{city_name}, {state}" if state else city_name
    params = {"address": address, "key": api_key}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["status"] == "OK":
            result = data["results"][0]
            location = result["geometry"]["location"]

            # Check if the city is in Alaska
            is_in_alaska = False
            for component in result.get("address_components", []):
                if "administrative_area_level_1" in component.get("types", []):
                    if component.get("short_name") == "AK":
                        is_in_alaska = True
                        break

            full_location = result.get("formatted_address", city_name)
            return location["lat"], location["lng"], is_in_alaska, full_location
        else:
            logger.error(f"Geocoding error for {city_name}: {data['status']}")
            return None, None, False, None
    except Exception as e:
        logger.error(f"Error fetching coordinates for {city_name}: {e}")
        return None, None, False, None


def get_weather_forecast(city_name, check_alaska=False):
    """
    Fetches the current temperature and short forecast for a city.

    Args:
        city_name: Name of the city
        check_alaska: If True, will check if the city is in Alaska

    Returns:
        Dictionary with weather information and Alaska status
    """
    lat, lon, is_in_alaska, full_location = get_lat_lon(city_name)
    if not lat or not lon:
        return {
            "city": city_name,
            "temp": "--",
            "condition": "Location not found",
            "is_in_alaska": False,
            "full_location": city_name,
        }

    headers = {"User-Agent": "(myweatherapp.com, contact@myweatherapp.com)"}

    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        point_resp = requests.get(points_url, headers=headers)
        point_resp.raise_for_status()
        point_data = point_resp.json()

        forecast_url = point_data["properties"]["forecast"]
        forecast_resp = requests.get(forecast_url, headers=headers)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()

        current_period = forecast_data["properties"]["periods"][0]

        return {
            "city": city_name,
            "temp": f"{current_period['temperature']}°{current_period['temperatureUnit']}",
            "condition": current_period["shortForecast"],
            "icon": current_period.get("icon"),
            "is_in_alaska": is_in_alaska,
            "full_location": full_location,
        }

    except Exception as e:
        logger.error(f"Error fetching weather for {city_name}: {e}")
        return {
            "city": city_name,
            "temp": "--",
            "condition": "Error fetching data",
            "is_in_alaska": is_in_alaska,
            "full_location": full_location or city_name,
        }


def get_alaska_weather():
    """
    Fetches weather for the top 5 Alaskan cities.
    """
    cities = ["Anchorage", "Fairbanks", "Juneau", "Wasilla", "Sitka"]
    results = []
    for city in cities:
        results.append(get_weather_forecast(city))
    return results
