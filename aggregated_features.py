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
import re
import h3
from sklearn.cluster import DBSCAN
import plotly.graph_objs as go


def time_filtration(converted_interactions, threshold):
    """
    Removes interactions if too close in time for it may impact the culstering process

    Arguments:
    converted_interactions: pandas DataFrame
                            Contains CDR data after being passed through preprocessing
                            It is assumed that the timestamp column is named 'end_absolute_time'
    threshold: int
               Number of minutes under which the difference between two interactions is to close and the second interaction is filtered out
    
    Returns a pandas dataframe with the same columns as the input one
    """
    converted_interactions['delta_time'] = converted_interactions['end_absolutetime_ts'].diff()
    return (converted_interactions
            .loc[lambda df: (df.delta_time > pd.Timedelta(f'{threshold} minutes')) | (df.delta_time < pd.Timedelta('0 minutes'))]
            .drop('delta_time', axis=1)
            .reset_index(drop= True))

def perform_clustering(filtered_interactions, eps, min_samples):
    """
    Applies spatial clustering to filtered interactions for each day. Allows to further identify trips and destination patterns in the IMSI mobility.

    This function takes as input a DataFrame containing filtered interactions, as well as the `eps` and `min_samples` parameters for DBSCAN clustering.
    It applies separate spatial DBSCAN clustering for each day on 'x' and 'y'.
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed together,
    while marking points that are alone in low-density regions as outliers. It requires two parameters: `eps`,
    which controls the maximum distance between two samples for them to be considered as in the same neighborhood, and `min_samples`,
    which controls the minimum number of samples to create a new neighborhood.
    Next, the function merges consecutive clusters by updating the cluster labels and recalculating the cluster sizes.
    The resulting DataFrame is returned with updated cluster labels and cluster sizes.

    Arguments:
    filtered_interactions: pandas.DataFrame
                                  DataFrame containing preprocessed and filtered CDR interactions.
    eps: float
        The `eps` parameter for DBSCAN clustering. This parameter controls the maximum distance between two samples for them to be considered as in the same neighborhood. 
    min_samples: int
                 The `min_samples` parameter for DBSCAN clustering. This parameter controls the minimum number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns: pandas.DataFrame
             DataFrame containing the interactions with updated cluster labels and cluster sizes.
    """

    #classical parameters: eps = 2700, min_sample = 4
    dates = filtered_interactions.groupby('start_date')
    list_date_df = []
    cluster_count = 0
    #Apply distinct clustering to each day
    for _, date_df in dates:
        # Extract 'x' and 'y' columns from the DataFrame
        data = date_df[['x', 'y']].values

        # Create DBSCAN object
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # Fit the clustering model to the data
        dbscan.fit(data)

        # Retrieve the cluster labels assigned by DBSCAN
        labels = dbscan.labels_

        # Retrieve the number of clusters and their respective sizes
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_sizes = np.bincount(labels + 1)

        # Add the cluster labels and sizes to the DataFrame
        date_df['cluster_label'] = np.where(labels != -1, labels + cluster_count, -1)
        date_df['cluster_size'] = np.where(labels != -1, cluster_sizes[labels + 1], 0)
        cluster_count += n_clusters
        list_date_df.append(date_df)
    
    labelled_interactions = pd.concat(list_date_df)

    #fusionate subsquent clusters

    labelled_interactions['updated_label'] = labelled_interactions['cluster_label']

    # Compare the current cluster label with the previous row's cluster label
    mask = ((labelled_interactions['cluster_label'] != -1)
             & (labelled_interactions['cluster_label'].shift() != -1)
               & (labelled_interactions['cluster_label'].shift() <= labelled_interactions['cluster_label'] -1))
    #drop dpulicates in the mask
    intervals = np.sort([*set(labelled_interactions.loc[mask].cluster_label.to_list())])
    apply_digitize = lambda x: np.digitize(x, intervals)
    labelled_interactions['updated_label'] = (labelled_interactions['cluster_label']
                                               - labelled_interactions['cluster_label']
                                               .apply(apply_digitize))

    dictionary_new_sizes = labelled_interactions.groupby('updated_label').count().hashed_imsi.to_dict()
    labelled_interactions['updated_size'] = labelled_interactions['updated_label'].map(dictionary_new_sizes)

    return labelled_interactions.drop(['cluster_label', 'cluster_size'], axis=1)

def display_clustering(interactions, hashed_imsi, eps, min_points, time_filtration_max):
    """
    Displays the output of a clustering on CDR interactions that distinguish the trips from the destinations.

    Parameters:
    interactions (pandas.DataFrame): A DataFrame containing preprocessed (and ideally filtered) CDR interactions.
    hashed_imsi (str): The hashed IMSI of the user.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_points (int): The number of samples in a neighborhood for a point to be considered as a core point.
    time_filtration_max (int): The maximum time difference between two consecutive interactions.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure containing the output of the clustering.
    """

    fig = go.Figure()

    imsi_interactions = interactions.loc[lambda df: df.hashed_imsi == f'{hashed_imsi}']
    filtered_interacciones = time_filtration(imsi_interactions, time_filtration_max)
    interactions_clustered = perform_clustering(filtered_interacciones, eps, min_points)
    dates = interactions_clustered.groupby('start_date')
    for dates, dates_df in dates:
        fig.add_trace(go.Scatter(x=dates_df['x'], y=dates_df['y'], mode='markers', name=dates, marker=dict(color=dates_df['updated_label'], colorscale='Viridis'), hovertemplate='label: %{marker.color}<br>timestamp: %{text}'))
        fig.update_traces(text=dates_df['end_absolutetime_ts'])
        fig.update_layout(height=600, width=1000, title_text="Interactions groupbe by trip and destinations")
    return fig

def compute_h3_number(interactions, resolution):
    """
    Issue the number of H3 resolutions 7 where an IMSI has at least one interaction, reflect how much moved the IMSI spatially.
    Parameters:
    interactions (pandas.DataFrame): A DataFrame containing preprocessed (and ideally filtered) CDR interactions.
    resolution (int): the h3 resolution (from 1 to 16) you want to take as a scale for the imsi's spatial expansion.
                      Resolution 7 is usually used.

    Returns: int
             The number of h3 where the IMSI has at least one interaction for the given resolution
    
    """
    latitudes = interactions['latitude'].values
    longitudes = interactions['longitude'].values
    h3_indexes = []
    for lat, lon in zip(latitudes, longitudes):
        h3_indexes.append(h3.geo_to_h3(lat.item(), lon.item(), resolution=resolution))
    return np.unique(h3_indexes).shape[0]

def compute_speed(converted_interactions, filtration_outlier_threshold=80):
    """
    Add instantaneous speed and filter outliers according to speed

    Parameters:
    -----------
    converted_interactions : pandas.DataFrame
        A dataframe containing the following columns:
            - end_absolutetime_ts : datetime64[ns]
            - x : float
            - y : float

    filtration_outlier_threshold : float, optional (default=80)
        The threshold value for filtering out instantaneous speed outliers. Any point with an in-speed and an out-speed superior to this threshold is filtered out.

    Returns:
    --------
    pandas.DataFrame
        A filtered dataframe containing the following columns:
            - end_absolutetime_ts : datetime64[ns]
            - x : float
            - y : float
            - instantaneous_speed : float
            
    """
    converted_interactions_copy = converted_interactions.copy()
    converted_interactions_copy['end_absolutetime_ts'] = pd.to_datetime(converted_interactions_copy['end_absolutetime_ts'])
    converted_interactions_copy['deltat'] = converted_interactions_copy['end_absolutetime_ts'].diff()
    converted_interactions_copy['deltal'] = (abs(converted_interactions_copy['y'].diff()) + abs(converted_interactions_copy['x'].diff()))

    converted_interactions_copy['deltat'] = converted_interactions_copy['deltat'].where(lambda x: x >= pd.Timedelta(0))
    converted_interactions_copy['instantaneous_speed'] = converted_interactions_copy['deltal'] / converted_interactions_copy['deltat'].dt.total_seconds()

    if filtration_outlier_threshold is not None:
        mask = (converted_interactions_copy['instantaneous_speed'] > filtration_outlier_threshold) & (converted_interactions_copy['instantaneous_speed'].shift(-1) > filtration_outlier_threshold)
        converted_interactions_copy = converted_interactions_copy[~mask]

    converted_interactions_copy = converted_interactions_copy.query(f'instantaneous_speed != {np.inf}') #filtration of outliers also applies to inf

    return converted_interactions_copy.drop(['deltat', 'deltal'], axis=1)
