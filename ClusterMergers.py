import tensorflow as tf
import numpy as np
from astropy.io import fits
import os


class BaseModel():
    
    def __init__(self):
        self.model = tf.keras.models.load_model('./model')
    
    def predict_merger_prob(self, X, normalise=True):
        """
        Inputs: 
        X: Numpy array with dimensions (N, 96, 96, 2):
            - Channel 1 is simulated Compton-y map
            - Channel 2 is simulated X-ray flux (0.1-15 keV)
            
        normalise: 
        True (default) - images are normalised using model training data within the prediction process
        False - images are pre-normalised outside of model prediction, recommended if simulation data is not generated in the pipeline used for the study, can use normalise function if needed
        
        Output: An array with each element corresponding to the input cluster merger probability
        """
        sz_input = X[:,:,:,0].reshape((-1,96,96,1))
        xr_input = X[:,:,:,1].reshape((-1,96,96,1))
        
        if not normalise:
            norm_layer_sz = self.model.get_layer("normalization_4") # SZ normalisation layer
            norm_layer_xr = self.model.get_layer("normalization_5") # Xray normalisation layer
            norm_model = tf.keras.Model(self.model.inputs, [norm_layer_sz.output, norm_layer_xr.output])
            new_model = tf.keras.Model(inputs=[norm_layer_sz.output, norm_layer_xr.output], outputs=self.model.layers[-1].output)
            y_prob = new_model.predict([sz_input, xr_input])
        
        else:
            y_prob = self.model.predict([sz_input, xr_input])
        
        return y_prob
        
    def read_fits(self, file_loc='example.fits'):
        print('To do')
        
        
        