import pandas as pd


# merge weather and turbine data into one dataset
def merge_data(df1,df2):
    df3 = df1.merge(df2,how='inner')
    return df3




# Cleaning Outliers in features save the xlong and ylat variables
def preprocess_data(df):
    clean_df = df.copy()
    drop_xy = clean_df.drop(['xlong','ylat'],axis=1)

    for feature in drop_xy.columns:
        Q1 = df[feature].quantile(.25)
        Q3 = df[feature].quantile(.75)

        IQR = Q3 - Q1

        low_bound = Q1 - (1.5 * IQR)
        high_bound = Q3 + (1.5 * IQR)

        clean_df = clean_df[(clean_df[feature] >= low_bound)&(clean_df[feature] <= high_bound)]

    
    clean_df = clean_df.drop(['t_rsa','t_ttlh'],axis=1)

    clean_df.to_csv('data/processed/fin_data.csv')





