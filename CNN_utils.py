import pandas as pd
import numpy as np
import plotly
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
pd.options.plotting.backend = 'plotly'
from scipy.spatial.distance import cdist
from tqdm.notebook import trange, tqdm
from sklearn.cluster import DBSCAN

#In this file are the functions necessary to run the pipeline

#Give the earth coordinates of the origin of the orthonormal coordinate system that is going to be implemented next
lon0, lat0 = -71.75, -34.19


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)

    Arguments:
    lon1: float that represent the longitude of the first point 
    lat1: float that represent the latitude of the first point 
    lon2: float that represent the longitude of the second point 
    lat2: float that represent the latitude of the second point 

    Returns a floating number, the distance in meters
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000 # Radius of earth in meters. Use 3956 for miles. Determines return value units.
    return c * r

def end_geo_treatment_lon(stringg):
    """
    Converts 'end_geo' column from CDR database into longitude
    Takes as argument a string that locate a point according to the earth universal coordinate system uner the format 'POINT(longitude latitude)'
    Returns the longitude as a string
    """
    separation = stringg.split('(')[1][:-1].split(' ')
    return separation[0]

def end_geo_treatment_lat(stringg):
    """
    Converts 'end_geo' column from CDR database into latitude
    Takes as argument a string that locate a point according to the earth universal coordinate system under the format 'POINT(longitude latitude)'
    Returns the latitude as a string    
    """
    separation = stringg.split('(')[1][:-1].split(' ')
    return separation[1]


def preprocess(interactions, lon0, lat0):
    """
    Converts the raw CDR database into a processable dataframe
    First converts the string encapsulating Earth coordinates into longitude (float) and latitude (float)
    Then creates an orthonormal coordinate system and add two columns with the cartesian coordinate ('x','y') of each interaction
    Drops the column of raw Earth coordinates

    Arguments:
    interacions: pandas dataframe
                 Contains the interactions that are to be treated further under the format
                 Index  hashed_imsi	 start_date  end_absolutetime_ts  end_geo
                 Each row represent an interaction between a mobile device and a cell tower
                 Index (int) number of the raw
                 hashed_imsi (string) column contains the International Mobile Suscriber Identities (IMSI), an universal set of identifiers for all connected devices 
                 start_date column (string) contains the date of the interaction under the format 'year-month-day'
                 end_absolutetime_ts (string) contains the complete timestamp of the interaction under the format 'year-month-day hour:minute:second'
                 end_geo (string) contains the coordinates of the device according to the Earth Coordinate System. They are given under the format 'POINT(longitude latitude)'
                 It is assumed that the columns will have these names

    lon0: float
          Longitude for the origin of the orthonormal coordinate system to be implemented
    lat0: float
          Latitude for the origin of the orthonormal coordinate system to be implemented

    Returns a pandas dataframe containing longitude, latitude and the coordinates 'x' and 'y' of each interaction in the new orthonormal system

    """
    interactions_copy = interactions.copy()
    interactions_copy['longitude'] = interactions_copy['end_geo'].apply(end_geo_treatment_lon)
    interactions_copy['latitude'] = interactions_copy['end_geo'].apply(end_geo_treatment_lat)
    sorted_interactions = (interactions_copy
                           .astype({'longitude': 'float64', 'latitude': 'float64'})
                           .sort_values(['hashed_imsi', 'end_absolutetime_ts' ])
                           .reset_index(drop = True)
                           )
    sorted_interactions['end_absolutetime_ts'] = pd.to_datetime(sorted_interactions['end_absolutetime_ts'])
    converted_interactions = (sorted_interactions
                              .assign(x= lambda df: np.sign(df.longitude - lon0)*haversine(lon0,0,df.longitude,0))
                              .assign(y= lambda df: np.sign(df.latitude - lat0)*haversine(0,lat0,0,df.latitude))
                              .drop(['end_geo'], axis=1))
                              
    return converted_interactions

