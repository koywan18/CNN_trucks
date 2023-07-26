import pandas as pd
import numpy as np
from tqdm.notebook import trange, tqdm

from CDR_preprocessing import *
from aggregated_features import *


def heatmap_generation(interactions, IMSI, max_x = 260, max_y = 277, unit=600, max_quantile_scaler = 92):
    """
    This function generates a 2D black and white heatmap of an IMSI's interactions
    Parameters:
    -----------
    interactions : pandas.DataFrame
        A dataframe containing the interactions during trips/destinations for an IMSI, sorted by timestamps,  with the following column:
            - x : float
            - y : float
    IMSI : string
        The hashed imsi identifying the mobile device
    max_x : int
        The maximum value of the x (converted longitude) dimension in the array
    max_y : int
        The maximum value of the y (converted latitude) dimension in the array
    unit: int
        The dimension (in meter) of a unit square in the heatmap
    max_quantile_scaler : int between 1 and 100
        The quantile cap to rescale the heatmap and give more weight to points with fewer interactions
    Returns:
    --------
    numpy.array
        A numpy array where each coordinate corresponds to the rescaled number of interactions in a square of dimension unit*unit 

    """
    interactions_df = interactions.loc[IMSI].copy()
    interactions_df['y_index'] = interactions_df['y'] / unit #600m is the new unit
    interactions_df['x_index'] = interactions_df['x'] / unit
    filtered_df = interactions_df.loc[lambda df: (df['x_index'] < max_x) & (df['y_index'] < max_y)]
    if filtered_df.empty: #if an IMSI is out of the map or unsignificant
        return None
    filtered_df['y_index'] = filtered_df['y_index'].astype(int)
    filtered_df['x_index'] = filtered_df['x_index'].astype(int)
    indexed_df = filtered_df.assign(norm_index=lambda df: df['x_index'].astype(str) + '_' + df['y_index'].astype(str))
    count_df = indexed_df.groupby('norm_index').size().reset_index(name='count')
    count_df[['x', 'y']] = count_df['norm_index'].str.split('_', expand=True)
    count_df['x'] = count_df['x'].astype(int)
    count_df['y'] = count_df['y'].astype(int)
    x_range = range(max_x)
    y_range = range(max_y)
    array = np.zeros((len(y_range), len(x_range)))
    for _, row in count_df.iterrows():
        x = row['x']
        y = row['y']
        count = row['count']
        array[y, x] = count

    #rescaling of the array: percentile 92 without 0 is taken as max value and rescaled on 0-255
    nonzero_indices = np.nonzero(array)
    # Create a new array without zeros
    array_without_zeros = array[nonzero_indices]
    perc = np.percentile(array_without_zeros, max_quantile_scaler)
    scaled_array = np.round(np.clip(array, None, perc)*255/perc)
    return scaled_array

def input_pipeline(interactions, label_dictionnary, h3_resolution = 7, eps = 2700, min_points = 4,
                    time_filtration_max  = 1, max_x = 260, max_y = 277, unit=600, max_quantile_scaler = 92):
    """
    This function computes the whole pipeline to transform an interaction dataframe into an input for the CNN.
    This input is a numpy array where are concatenated a heatmap of the IMSI interactions and a set of aggregated features.
    First the interactions are preprocessed
    Then they are fed to the aggregated features pipeline which issue a set of additionnal features
    Then for each imsi, the heatmap is generated and concatenated with the aggregated features 
    Finally the resulting arrays are stored in a dataframe

    Parameters:
    -----------
    interactions : pandas.DataFrame
        A dataframe containing the interactions during trips/destinations for an IMSI, sorted by timestamps,  with the following column:
            - start_date : datetime64[ns]; the date of each interaction
            - hashed_imsi : string; the international identifier of each mobile device
            - end_geo : object; contains the earth coordinates of each interaction
            - end_absolutetime_ts : datetime64[ns]; contains the timestamps of each interaction
    label_dictionnary : dict
        A dictionary with all imsi and their label (1: truck, 0: non truck)
    h3_resolution: int between 1 and 16
        The resolution of the H3 grid used to compute the number of H3 cells per day
    eps: int
        The maximum distance between two samples for them to be considered as part of the same cluster for DBSCAN clustering
    min_points: int
        The number of samples in a neighborhood for a point to be considered as a core point for DBSCAN clustering
    time_filtration_max: int
        The minimum time difference between two interactions in minutes
    max_x : int
        The maximum value of the x (converted longitude) dimension in the array
    max_y : int
        The maximum value of the y (converted latitude) dimension in the array
    unit: int
        The dimension (in meter) of a unit square in the heatmap
    max_quantile_scaler : int between 1 and 100
        The quantile cap to rescale the heatmap and give more weight to points with fewer interactions
    Returns:
    --------
    pandas.DataFrame
        A pandas Dataframe where are stored the array concatenating the heatmap and aggregated features as well as the label of each IMSI.
    """
    imsi_array_list = []
    imsi_out_of_map = []
    clean_interactions = preprocess(interactions.drop_duplicates())
    output_aggregated_features = apply_pipeline(clean_interactions,
                                                 h3_resolution,
                                                 eps,
                                                 min_points,
                                                 time_filtration_max)['Variables']
    variable_number = len(output_aggregated_features[0])
    df_to_complete = (pd.DataFrame([[[0]*(max_x - variable_number)]*output_aggregated_features.shape[0]])
                      .T.set_index(output_aggregated_features.index)
                      .rename(columns =  {0 : 'Variables'}))
    aggregated_data_completed_by_zero = output_aggregated_features.to_frame() + df_to_complete
    imsi_list = aggregated_data_completed_by_zero.index.to_list()

    for imsi in tqdm(imsi_list):
        aggregated_data_imsi = aggregated_data_completed_by_zero.loc[imsi, 'Variables']
        array_imsi = heatmap_generation(clean_interactions.set_index('hashed_imsi'),
                                        imsi,
                                        max_x,
                                        max_y,
                                        unit,
                                        max_quantile_scaler)
        if array_imsi is None:
            imsi_out_of_map.append(imsi)
        else:
            imsi_array_list.append(np.concatenate([array_imsi, np.array([aggregated_data_imsi])]))

    imsi_list_in = list(filter(lambda x: x not in imsi_out_of_map, imsi_list))
    input_dataframe = (pd.DataFrame({'hashed_imsi' : imsi_list_in, 'array': imsi_array_list})
                       .assign(label = lambda df: df.hashed_imsi.map(label_dictionnary)))
    return input_dataframe