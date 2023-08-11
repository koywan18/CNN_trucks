# CNN_trucks
This repository introduces a classification algorithm for CDR data in order to identify high load trucks

## Data
CDR data is collected by network connection providers whenever a suscriber makes an internet request: looking for a website on google, sending a whatsapp message, etc. The internet provider saves the timestamp and triangulates the device using its network cell towers. These informations are associated to the International Mobile Suscriber Identity (IMSI), a unique hash representing the device.
This results in data similar to GPS points. However CDR data has less accuracy and irregulate time frequency. Moreover, typical effects of cell towers, such as the ping pong effect, are known to generate outliers in the data. Hence the data must be preprocessed before any classification attempts.

## Methodology
Preprocessed CDR data is passed through a supervised deep-learning algorithm. More specifically, a CNN is fed with a blend of raw and aggregated features to issue predictions for each IMSI.
This allows to capture both geographical and mobility patterns.

### Aggregated features
A set of aggregated features is computed to illustrate the mobility of each IMSI and feed it into the neural network. Special attention is paid to distinguishing between road and urban user behavior, thanks to a previous clustering. The features are stated as follows:
•	Destination number per day
•	Mean speed during trips
•	Mean speed within destination
•	Mean duration of destinations
•	Number of H3 resolution 7 where there is at least one interaction
•	Number of interactions during a trip per day
•	Number of interactions within a destination per day
•	Number of interactions per day
•	Mean duration of destinations per day
•	Mean duration of trip per day

### Raw data heatmap
IMSI's interactions in the interest zone are summed up in a heatmap. The unit of the latter was set to 600m in the study for it allowed to reduce the dimension effectively without loosing key informations.

### Neural Network architecure
Both heatmap and aggregated features are concatenated in a tensor that is fed to the neural network. First the heatmap is processed through Convolutionnal and Max Pooling Layers in order to extract the geographical features and reduce the dimension. The geographical features are then concatenated to the aggregated features and the resulting vector is passed, after a Relu activation, through a linear classifier.

### Results
After an approximately 40-epoch training, the model converges and performs as follows on the test set:
Accuracy: 0.9297 on average (15 training on random set distribution)
f1_score: 0.8861 on average (15 training on random set distribution)
