# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                            SamplingLayer
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)

import keras
import torch
from torch.distributions.normal import Normal

# Note : https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class SamplingLayer(keras.layers.Layer):
    '''A custom layer that receive (z_mean, z_var) and sample a z vector'''

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        
        batch_size, latent_dim = z_mean.shape
        
        epsilon = Normal(0, 1).sample((batch_size, latent_dim)).to(z_mean.device)

        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon 
        
        return z