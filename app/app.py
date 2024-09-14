from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os 

app = Flask(__name__)

# Load the model
model = pickle.load(open('models/treereg.pkl', 'rb'))

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_data(latitude, longitude):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2023-11-01",
        "end_date": "2024-05-31",
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant"]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(1).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(2).ValuesAsNumpy()

    daily_data = {
        "temperature_2m_mean": daily_temperature_2m_mean,
        "wind_speed_10m_max": daily_wind_speed_10m_max,
        "wind_direction_10m_dominant": daily_wind_direction_10m_dominant
    }

    df = pd.DataFrame(data=daily_data)
    
    weather_stats = {
        "temp_avg": df["temperature_2m_mean"].mean(),
        "temp_std": df["temperature_2m_mean"].std(),
        "wind_speed_avg": df["wind_speed_10m_max"].mean(),
        "wind_speed_std": df["wind_speed_10m_max"].std(),
        "wind_direction_avg": df["wind_direction_10m_dominant"].mean(),
        "wind_direction_std": df["wind_direction_10m_dominant"].std()
    }
    
    return weather_stats

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        year = int(request.form['year'])
        hub_height = float(request.form['hub_height'])
        rotor_diameter = float(request.form['rotor_diameter'])
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])

        # Get weather data
        weather_stats = get_weather_data(latitude, longitude)

        # Prepare input for the model
        input_data = np.array([[
            year,
            hub_height,
            rotor_diameter,
            longitude,
            latitude,
            weather_stats['temp_avg'],
            weather_stats['temp_std'],
            weather_stats['wind_speed_avg'],
            weather_stats['wind_speed_std'],
            weather_stats['wind_direction_avg'],
            weather_stats['wind_direction_std']
        ]])

        # Make prediction
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
