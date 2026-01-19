# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                         GAN / Generators
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)


import numpy as np
import torch.nn as nn


# -----------------------------------------------------------------------------
# -- Discriminator n°1
# -----------------------------------------------------------------------------
#
class Discriminator_1(nn.Module):
    '''
    A basic DNN discriminator, usable with classic GAN
    '''

    def __init__(self, latent_dim=None, data_shape=None):
    
        super().__init__()
        self.img_shape = data_shape
        print('init discriminator 1     : ',data_shape,' to sigmoid')

        self.model = nn.Sequential(

            nn.Flatten(),
            nn.Linear(int(np.prod(data_shape)), 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)

        return validity



# -----------------------------------------------------------------------------
# -- Discriminator n°2
# -----------------------------------------------------------------------------
#
class Discriminator_2(nn.Module):
    '''
    A more efficient discriminator,based on CNN, usable with classic GAN
    '''

    def __init__(self, latent_dim=None, data_shape=None):
    
        super().__init__()
        self.img_shape = data_shape
        print('init discriminator 2     : ',data_shape,' to sigmoid')

        self.model = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(12544, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_nchw = img.permute(0, 3, 1, 2) # reformat from NHWC to NCHW
        validity = self.model(img_nchw)

        return validity


   
# -----------------------------------------------------------------------------
# -- Discriminator n°3
# -----------------------------------------------------------------------------
#     
class Discriminator_3(nn.Module):
    '''
    A CNN discriminator, usable with a WGANGP.
    This discriminator has no sigmoid and returns a critical and not a probability
    '''

    def __init__(self, latent_dim=None, data_shape=None):
    
        super().__init__()
        self.img_shape = data_shape
        print('init discriminator 3     : ',data_shape,' to sigmoid')

        self.model = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(12544, 1),
            nn.Sigmoid(),

        )

    def forward(self, img):
        img_nchw = img.permute(0, 3, 1, 2) # reformat from NHWC to NCHW
        validity = self.model(img_nchw)

        return validity