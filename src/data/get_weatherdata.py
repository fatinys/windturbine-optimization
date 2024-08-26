import openmeteo_requests
import requests_cache
import pandas as pd
import os
import numpy as np
import time
from tqdm import tqdm
from retry_requests import retry
# General API call sourced from:
# https://open-meteo.com/en/docs/historical-weather-api#start_date=2023-11-01&end_date=2024-05-31&hourly=&daily=temperature_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant&timezone=America%2FChicago


# Free Rate limit on API
calls_per_turbine = 4.5
delay_between_turbines = 3.24  



#Splitting Data To deal with RateLimit
def split_df():

    path = os.path.join('data','interim', f"turbine_data.csv")
    turb_df = pd.read_csv(path)

    tloc_df = turb_df[['xlong','ylat']]


    split_data = np.array_split(tloc_df, 4)

    tdf1, tdf2, tdf3, tdf4 = split_data
    tdf1.to_csv('data/interim/turbinelocsplit/tdf1.csv', index=False)
    tdf2.to_csv('data/interim/turbinelocsplit/tdf2.csv', index=False)
    tdf3.to_csv('data/interim/turbinelocsplit/tdf3.csv', index=False)
    tdf4.to_csv('data/interim/turbinelocsplit/tdf4.csv', index=False)

####################################################
# *MUST RUN split_df() BEFORE RUNNING FETCH_WEATHER*
####################################################

# fetch_weather(tdf(specific split of the tloc), split_name("name of the output data"))
def fetch_weather(tdf,split_name):

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=10, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Date range for the data
    start_date = "2023-11-01"
    end_date = "2024-05-31"

    # Store results
    results = []

    # Iterate over the DataFrame rows
    for index, row in tqdm(tdf.iterrows(), total=len(tdf)):
        params = {
            "latitude": row['ylat'],
            "longitude": row['xlong'],
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant"],
            "timezone": "America/Chicago"
        }
        
        responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        response = responses[0]
        
        # Process daily data
        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(1).ValuesAsNumpy()
        daily_wind_direction_10m_dominant = daily.Variables(2).ValuesAsNumpy()
        
        # Compute statistics
        avg_temp = np.mean(daily_temperature_2m_mean)
        std_temp = np.std(daily_temperature_2m_mean)
        avg_wind_speed = np.mean(daily_wind_speed_10m_max)
        std_wind_speed = np.std(daily_wind_speed_10m_max)
        avg_wind_direction = np.mean(daily_wind_direction_10m_dominant)
        std_wind_direction = np.std(daily_wind_direction_10m_dominant)
        
        results.append({
            "latitude": row['ylat'],
            "longitude": row['xlong'],
            "avg_temp": avg_temp, 
            "std_temp": std_temp,
            "avg_wind_speed": avg_wind_speed,
            "std_wind_speed": std_wind_speed,
            "avg_wind_direction": avg_wind_direction,
            "std_wind_direction": std_wind_direction
        })
        
        time.sleep(3.24)


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    path = os.path.join('data', 'raw','weatherlocsplit', f"{split_name}.csv")
    results_df.to_csv(path)


def combine_wdf():

    wdf1 = pd.read_csv('data/raw/weatherlocsplit/wdf1.csv')
    wdf2 = pd.read_csv('data/raw/weatherlocsplit/wdf2.csv')
    wdf3 = pd.read_csv('data/raw/weatherlocsplit/wdf3.csv')
    wdf4 = pd.read_csv('data/raw/weatherlocsplit/wdf4.csv')
        
    df = pd.concat([wdf1,wdf2,wdf3,wdf4])

    df.to_csv('data/interim/weatherloc_stats.csv')

def combine_all():
    #path = os.path.join('data', 'interim', "turbine_data")
    df1 = pd.read_csv("data/interim/turbine_data.csv")
    #path = os.path.join('data', 'interim', "weatherloc_stats")
    df2 = pd.read_csv("data/interim/weatherloc_stats.csv")
    df = df1.merge(df2,how='inner')
    #path = os.path.join('data', 'interim', "weatherloc_stats.csv")
    df.to_csv("data/processed/fin_data.csv")