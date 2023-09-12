import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
from PIL import Image
import cv2

class BaseModel():
    """Methods associated with predicting whether galaxy clusters are merging."""
    
    def __init__(self):
        self.model = tf.keras.models.load_model('./model')
    
    def predict_merger_prob(self, X, normalise=True):
        """
        Function to return predictions on a set of images.
        
        Inputs
        --------
        X: numpy array - dimensions (N, 96, 96, 2)
            - channel 1 is simulated Compton-y map
            - channel 2 is simulated X-ray flux (0.1-15 keV)
            
        normalise: bool, optional
            True (default) - images are normalised using model training data within the prediction process
            False - images are pre-normalised outside of model prediction, recommended if simulation data 
            is not generated in the pipeline used for the study, can use normalise function if needed
        
        Output
        --------
        numpy array with each element corresponding to the input cluster merger probability, dimensions (N,1)
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
    
    def normalise_img(self, X):
        """
        Inputs
        --------
        X: numpy array - dimensions (N, 96, 96, 2)
            - channel 1 is simulated Compton-y map
            - channel 2 is simulated X-ray flux (0.1-15 keV)
            
        Output
        --------
        As input, but with each channel independently normalised.
        
        """
        
        mean_sz = np.mean(X[:,:,:,0])
        std_sz = np.std(X[:,:,:,0])
        mean_xr = np.mean(X[:,:,:,1])
        std_xr = np.std(X[:,:,:,1])
        
        X[:,:,:,0] = (X[:,:,:,0] - mean_sz) / std_sz
        X[:,:,:,1] = (X[:,:,:,1] - mean_xr) / std_xr
        
        return X
    

    def read_fits(self, folder_loc='example_folder', resize=True):
        """Function to read in sz and xray fits files and convert to numpy array.
        Inputs
        --------
        folder_loc: str
            Name of folder which contains two fits files to be combined:
            - 'SZ.fits'
            - 'Xray.fits'
            
        resize: bool, optional
            True: resizes to 96x96 (use if original image size is different to 96x96)
            False: keeps original image shape
            
        Output
        --------
        numpy array - dimensions (N, 96, 96, 2)
            - channel 1 is simulated Compton-y map
            - channel 2 is simulated X-ray flux (0.1-15 keV)
        
        """
        hdu = fits.open(folder_loc + '/SZ.fits', ignore_missing_simple=True)
        sz_img = hdu[0].data
        sz_img = np.array(Image.fromarray(sz_img))
        hdu.close()
        
        hdu = fits.open(folder_loc + '/Xray.fits', ignore_missing_simple=True)
        xr_img = hdu[0].data
        xr_img = np.array(Image.fromarray(xr_img))
        hdu.close()
        
        if resize:
            sz_img = cv2.resize(sz_img, dsize=(96, 96), interpolation=cv2.INTER_AREA)
            xr_img = cv2.resize(xr_img, dsize=(96, 96), interpolation=cv2.INTER_AREA)
                     
        X = np.stack([sz_img, xr_img], axis=2)
        npx = X.shape[1]
        X = X.reshape((-1,npx,npx,2))
        
        return X

        
        