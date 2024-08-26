import pandas as pd
import requests 
import os


# State parameter abbreviated("TX, CO, NY")

def turbine_get(state, lim=10000):
    #Base URL for Request
    url = "https://eersc.usgs.gov/api/uswtdb/v1/turbines"

    #Request Parameter
    param = {
    't_state': f'eq.{state}',
    'limit': lim
    }

    #Returning Dataframe
    response = requests.get(url, params=param)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    
    else:
        print(f"Failed to retrieve data: {response.status_code}")

# Create Turbine CSV
def turbine_csv(state, df):
    path = os.path.join('data', 'raw', f"{state}_turbines_raw.csv")
    df.to_csv(path, index=False)

# Preprocess Raw Turbine Data


def turbine_pp(df):

    # Get rid of useless variables
    variables = ['p_year','t_cap','t_hh','t_rd','t_rsa','t_ttlh','offshore','t_conf_atr','t_conf_loc','xlong','ylat']
    turbine_df = df[variables]

    # Get rid of offshore turbines
    turbine_df = turbine_df[turbine_df['offshore']!=1]

    # Consider only the turbines with high confidence data
    turbine_df = turbine_df[(turbine_df['t_conf_atr'] == 3) & (turbine_df['t_conf_loc'] == 3)]

    # Drop offshore and confidence columns
    turbine_df = turbine_df.drop(['offshore', 't_conf_loc', 't_conf_atr'],axis=1)

    turbine_df.to_csv('data/interim/turbine_data.csv', index=False)




data = turbine_pp(turbine_get("TX"))

turbine_csv("TX", data)

