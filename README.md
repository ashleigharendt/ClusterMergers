# ClusterMergers
This repository includes code to predict likelihood of a cluster merger, based on simulated SZ and X-ray images.     
It takes an array of 96 x 96 pixel images of clusters, with one channel corresponding to the SZ filter, and one for the X-ray, and outputs the likelihood of that cluster being a merger.     
The output is a probability taking a value between 0 and 1, with 1 being 100% likely to be merging.    
More information on the project and model is described in **link to paper pending**


### Dependencies
tensorflow == 2.12.0    
numpy == 1.23.5     
astropy == 5.2.1    
pillow == 9.4.0    
opencv == 4.7.0.72

## Repository items
*ClusterMergers.py* - an executable python file containing the model class needed for predictions. It contains methods to preprocess the images, if needed, as well as call the model to make predictions.    
*PredictionExample.ipynb* - contains an example of how to use the model to run a prediction.    
*example_folder/* - contains two example fits files taken from The 300 project simulations.    
*model/* - contains the tensorflow model ready to load and use.

## How to run
To make a single prediction, import and call BaseModel.predict_merger_prob() function within ClusterMergers.py, as shown in PredictionExample.ipynb.

## Contact information
For any questions please contact the author Ashleigh Arendt: [arendtash@hotmail.com](arendtash@hotmail.com). 

