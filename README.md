# CNN_trucks
This repository introduces a classification algorithm for CDR data in order to identify high load trucks

## Data
CDR data is collected by network connection providers whenever a suscriber makes an internet request, for example looking for a website on google, sending a whatsapp message, etc. The internet provider saves the timestamp and triangulates the device using its network cell towers. This information is associated to the International Mobile Suscriber Identity (IMSI), an unique hash representing the device.
This results in data similar to GPS points. However CDR data has less accuracy and an erratic time frequency. Moreover, typical effects of cell towers, such as the ping pong effect, are known to generate outliers in the data. Hence the data must be preprocessed before any classification attempts.

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

### Creation of a calibration set
There was no labelled set available to train the model as the litterature is still poor on the subject. Consequently, a calibration set was manually created by visualising the itinerary of some IMSIs one by one, computing their average speed and identifying typical trucks stops (logistic zones, commercial ports, airports, etc). 

### Neural Network architecure
Both heatmap and aggregated features are concatenated in a tensor that is fed to the neural network. First the heatmap is processed through Convolutionnal and Max Pooling Layers in order to extract the geographical features and reduce the dimension. The geographical features are then concatenated to the aggregated features and the resulting vector is passed, after a Relu activation, through a linear classifier.

### Results
The calibration set is splitted randomly into a training set (75% of the data) and a test set (25% of the data).
The model converges after an approximately 40-epoch training.
In order to issue some results and compare them to comon reference models -logistic regression, random forest, support vector machine (SVM)- we performed 15 successive training and testing on aleatory calibration set splitting and took the average for each statistics.
Thereafter is displayed the comparison between models for various statistics:

 Statistics         | Accuracy | Precision | Recall | F1 score | AUC ROC |
|-------------------|----------|-----------|--------|----------|---------|
|Logistic regression|0.837     |0.818      |0.711   |0.761     |0.895    |
|Random forest      |	0.827  |0.833      |0.658   |0.735     |0.848    |
|SVM (kernel rbf)   |0.817     |0.732      |0.789   |0.759     |0.852    |
|Concatenated CNN   |0.933     |0.970      |0.842   |0.901     |0.922    |


