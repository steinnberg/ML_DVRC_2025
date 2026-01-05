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
# JL Parouty (mars 2024)

import keras
import torch

# See : https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class VariationalLossLayer(keras.layers.Layer):
   
    def __init__(self, loss_weights=[3,7]):
        super().__init__()
        self.k1 = loss_weights[0]
        self.k2 = loss_weights[1]


    def call(self, inputs):
        
        k1 = self.k1
        k2 = self.k2
        
        # ---- Retrieve inputs
        #
        x, z_mean, z_log_var, y = inputs
        
        # ---- Compute : reconstruction loss
        #                
        r_loss = torch.nn.functional.binary_cross_entropy(y, x, reduction='sum')
        
        #
        # ---- Compute : kl_loss
        # 
        kl_loss = - torch.sum(1+ z_log_var - z_mean.pow(2) - z_log_var.exp())
        
        # ---- Compute total loss, and add it
        #
        loss = r_loss*k1 + kl_loss*k2
        self.add_loss(loss)
        
        return y

    
    def get_config(self):
        return {'loss_weights':[self.k1,self.k2]}
    