# CNN_trucks
This repository introduces a classification algorithm for CDR data in order to identify trucks

## Data
CDR data is collected by network connection providers whenever a suscriber makes an internet request: looking for a website on google, sending a whatsapp message, etc. The internet provider saves the timestamp and triangulates the device using its network cell towers. This results in data similar to GPS points. However CDR data has less accuracy and irregulate time frequency. 

## Methodology
Preprocessed CDR data is passed through a supervised deep-learning algorithm. More specifically, a CNN is fed with a blend of raw and aggregated features to issue predictions for each IMSI.
This allows to capture both geographical and mobility patterns.

### Aggregated features
A set of aggregated features is computed to illustrate the mobility of each IMSI and feed it into the neural network. In order to do so, 
### Raw data heatmap
### Results
