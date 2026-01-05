# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                              VAE Example
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (mars 2024

import numpy as np
import keras
import torch

from IPython.display import display,Markdown
from modules.layers    import SamplingLayer
import os


# Note : https://keras.io/guides/making_new_layers_and_models_via_subclassing/



class VAE(keras.Model):
    '''
    A VAE model, built from given encoder and decoder
    '''

    version = '2.0'

    def __init__(self, encoder=None, decoder=None, loss_weights=[1,1], **kwargs):
        '''
        VAE instantiation with encoder, decoder and r_loss_factor
        args :
            encoder : Encoder model
            decoder : Decoder model
            loss_weights : Weight of the loss functions: reconstruction_loss and kl_loss
            r_loss_factor : Proportion of reconstruction loss for global loss (0.3)
        return:
            None
        '''
        super(VAE, self).__init__(**kwargs)
        self.encoder      = encoder
        self.decoder      = decoder
        self.loss_weights = loss_weights
        print(f'Fidle VAE is ready :-)  loss_weights={list(self.loss_weights)}')
       
        
    def call(self, inputs):
        '''
        Model forward pass, when we use our model
        args:
            inputs : Model inputs
        return:
            output : Output of the model 
        '''
        z_mean, z_log_var, z = self.encoder(inputs)
        output               = self.decoder(z)
        return output
                
        
    def train_step(self, input):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics are loss.
        args:
            inputs : Model inputs
        return:
            loss    : Total loss
            r_loss  : Reconstruction loss
            kl_loss : KL loss
        '''
        
        # ---- Get the input we need, specified in the .fit()
        #
        if isinstance(input, tuple):
            input = input[0]
        
        k1,k2 = self.loss_weights
        
        # ---- Reset grad
        #
        self.zero_grad()
        
        # ---- Forward pass
        #
        # Get encoder outputs
        #
        z_mean, z_log_var, z = self.encoder(input)
            
        # ---- Get reconstruction from decoder
        #
        reconstruction       = self.decoder(z)
        
        # ---- Compute loss
        #      Total loss = Reconstruction loss + KL loss
        #
        r_loss  = torch.nn.functional.binary_cross_entropy(reconstruction, input, reduction='sum')
        kl_loss = - torch.sum(1+ z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss    = r_loss*k1 + kl_loss*k2
        
        # ---- Compute gradients for the weights
        #
        loss.backward()
        
        # ---- Adjust learning weights
        #
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        
        # ---- Update metrics (includes the metric that tracks the loss)
        #
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(input, reconstruction)
        
        # ---- Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        #
        return {m.name: m.result() for m in self.metrics}
        
        
        
        
        # # ---- Forward pass
        # #      Run the forward pass and record 
        # #      operations on the GradientTape.
        # #
        # with tf.GradientTape() as tape:
            
        #     # ---- Get encoder outputs
        #     #
        #     z_mean, z_log_var, z = self.encoder(input)
            
        #     # ---- Get reconstruction from decoder
        #     #
        #     reconstruction       = self.decoder(z)
         
        #     # ---- Compute loss
        #     #      Reconstruction loss, KL loss and Total loss
        #     #
        #     reconstruction_loss  = k1 * tf.reduce_mean( keras.losses.binary_crossentropy(input, reconstruction) )

        #     kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        #     kl_loss = -tf.reduce_mean(kl_loss) * k2

        #     total_loss = reconstruction_loss + kl_loss

        # # ---- Retrieve gradients from gradient_tape
        # #      and run one step of gradient descent
        # #      to optimize trainable weights
        # #
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # return {
        #     "loss":     total_loss,
        #     "r_loss":   reconstruction_loss,
        #     "kl_loss":  kl_loss,
        # }
    
    
    def predict(self,inputs):
        '''Our predict function...'''
        z_mean, z_var, z  = self.encoder.predict(inputs)
        outputs           = self.decoder.predict(z)
        return outputs

        
    def save(self,filename):
        '''Save model in 2 part'''
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.keras')
        self.decoder.save(f'{filename}-decoder.keras')

    
    def reload(self,filename):
        '''Reload a 2 part saved model.'''
        filename, extension = os.path.splitext(filename)
        self.encoder = keras.models.load_model(f'{filename}-encoder.keras', custom_objects={'SamplingLayer': SamplingLayer})
        self.decoder = keras.models.load_model(f'{filename}-decoder.keras')
        print('Reloaded.')
                
        
    @classmethod
    def about(cls):
        '''Basic whoami method'''
        display(Markdown('<br>**FIDLE 2024 - VAE**'))
        print('Version              :', cls.version)
        print('Keras version        :', keras.__version__)
