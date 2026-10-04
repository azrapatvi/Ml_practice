import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# function to find the distance km betwenn restaurant ad delivery location
def calculate_distance(row):
    lat1 = radians(row['restaurant_latitude'])
    lon1 = radians(row['restaurant_longitude'])
    lat2 = radians(row['delivery_location_latitude'])
    lon2 = radians(row['delivery_location_longitude'])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return 6371 * c


def feature_engineering(df):

    df['festival'] = df['festival'].map({'Yes': 1, 'No': 0})

    df['time_order_picked_hr']=df['time_order_picked'].dt.hour
    df['time_order_picked_mins']=df['time_order_picked'].dt.minute
    df.drop('time_order_picked',axis=1,inplace=True)

    df['time_taken_mins']=df['time_taken(min)'].str.extract(r'(\d+)')
    df.drop('time_taken(min)',axis=1,inplace=True)

    df['time_taken_mins'] = pd.to_numeric(
    df['time_taken_mins'],
    errors='coerce'
    )

    df['time_orderd_hr']=df['time_orderd'].dt.hour
    df['time_orderd_mins']=df['time_orderd'].dt.minute

    df.drop('time_orderd',axis=1,inplace=True)

    df['orderd_day_of_week']=df['order_date'].dt.day_of_week
    df['orderd_month'] = df['order_date'].dt.month
    df['orderd_is_weekend']=df['orderd_day_of_week'].isin([5,6]).astype(int) # means 5->sat 6->sun so is weekend

    df.drop('order_date',axis=1,inplace=True)

    df['distance_km']=df.apply(calculate_distance, axis=1)

    df.drop(['restaurant_latitude','restaurant_longitude','delivery_location_latitude','delivery_location_longitude'],axis=1,inplace=True)

    
    return df



