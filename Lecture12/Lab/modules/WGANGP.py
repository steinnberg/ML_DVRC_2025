
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                   WGANGP LigthningModule
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)



import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lightning import LightningModule


class WGANGP(LightningModule):

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------
    #
    def __init__(
        self,
        data_shape          = (None,None,None),
        latent_dim          = None,
        lr                  = 0.0002,
        b1                  = 0.5,
        b2                  = 0.999,
        batch_size          = 64,
        lambda_gp           = 10,
        generator_name      = None,
        discriminator_name  = None,
        **kwargs,
    ):
        super().__init__()

        print('\n---- GAN initialization --------------------------------------------')

        # ---- Hyperparameters
        #
        # Enable Lightning to store all the provided arguments under the self.hparams attribute.
        # These hyperparameters will also be stored within the model checkpoint.
        #
        self.save_hyperparameters()

        print('Hyperarameters are :')
        for name,value in self.hparams.items():
            print(f'{name:24s} : {value}')

        # ---- Because we have more than one optimizer
        #
        self.automatic_optimization = False

        # ---- Generator/Discriminator instantiation
        #
        print('Submodels :')
        module=sys.modules['__main__']
        class_g = getattr(module, generator_name)
        class_d = getattr(module, discriminator_name)
        self.generator     = class_g( latent_dim=latent_dim, data_shape=data_shape)
        self.discriminator = class_d( latent_dim=latent_dim, data_shape=data_shape)

        # ---- Validation and example data
        #
        self.validation_z        = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)


    def forward(self, z):
        return self.generator(z)


    def adversarial_loss(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y)



    def gradient_penalty(self, real_images, fake_images):

        # see: https://medium.com/dejunhuang/implementing-gan-and-wgan-in-pytorch-551099afde3c

        batch_size = real_images.size(0)

        # ---- Create interpolate images
        #
        # Get a random vector : size=([batch_size])
        epsilon = torch.distributions.uniform.Uniform(0, 1).sample([batch_size])
        
        # Add dimensions to match images batch : size=([batch_size,1,1,1])
        epsilon = epsilon[:, None, None, None]
        
        # Put epsilon a the right place
        epsilon = epsilon.type_as(real_images)
        
        # Do interpolation
        interpolates = epsilon * fake_images + ((1 - epsilon) * real_images)

        # ---- Use autograd to compute gradient
        #
        # The key to making this work is including `create_graph`, this means that the computations
        # in this penalty will be added to the computation graph for the loss function, so that the
        # second partial derivatives will be correctly computed.
        #
        interpolates.requires_grad_()

        pred_labels = self.discriminator.forward(interpolates)

        gradients = torch.autograd.grad(  inputs       = interpolates,
                                          outputs      = pred_labels, 
                                          grad_outputs = torch.ones_like(pred_labels),
                                          create_graph = True, 
                                          retain_graph = True,
                                          only_inputs  = True )[0]

        grad_flat   = gradients.view(batch_size, -1)
        grad_norm   = torch.linalg.norm(grad_flat, dim=1)

        grad_penalty = (grad_norm - 1) ** 2 

        # gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

        return grad_penalty



    def training_step(self, batch, batch_idx):
        real_imgs  = batch
        batch_size = batch.size(0)
        lambda_gp  = self.hparams.lambda_gp


        optimizer_g, optimizer_d = self.optimizers()

        # ---- Get some latent space vectors
        #      We use type_as() to make sure we initialize z on the right device (GPU/CPU).
        #
        z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.type_as(real_imgs)
        
        # ---- Train generator ------------------------------------------------
        #      Generator use optimizer #0
        #      We try to generate false images that could mislead the discriminator
        # ---------------------------------------------------------------------
        #
        self.toggle_optimizer(optimizer_g)
                
        # Get fake images
        fake_imgs = self.generator.forward(z)
        
        # Get critics
        critics   = self.discriminator.forward(fake_imgs)

        # Loss
        g_loss = -critics.mean()

        # Log
        self.log("g_loss", g_loss, prog_bar=True)

        # Backward loss
        self.manual_backward(g_loss)
        
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        self.untoggle_optimizer(optimizer_g)

        # ---- Train discriminator --------------------------------------------
        #      Discriminator use optimizer #1
        #      We try to make the difference between fake images and real ones 
        # ---------------------------------------------------------------------
        #
        self.toggle_optimizer(optimizer_d)

        # Get critics
        critics_real = self.discriminator.forward(real_imgs)
        critics_fake = self.discriminator.forward(fake_imgs.detach())

        # Get gradient penalty
        grad_penalty = self.gradient_penalty(real_imgs, fake_imgs.detach())

        # Loss
        d_loss = critics_fake.mean() - critics_real.mean() + lambda_gp*grad_penalty.mean()

        # Log loss
        self.log("d_loss", d_loss, prog_bar=True)

        # Backward
        self.manual_backward(d_loss)
        
        optimizer_d.step()
        optimizer_d.zero_grad()
        
        self.untoggle_optimizer(optimizer_d)
 


    def configure_optimizers(self):

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # With a GAN, we need 2 separate optimizer.
        # opt_g = torch.optim.Adam(self.generator.parameters(),     lr=lr, betas=(b1, b2))
        # opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2),)
        opt_g = torch.optim.Adam(self.generator.parameters(),     lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []


    def on_train_epoch_end(self):

        # ---- Log Graph
        #
        if(self.current_epoch==1):
            sampleImg=torch.rand((1,28,28,1))
            sampleImg=sampleImg.type_as(self.generator.model[0].weight)
            self.logger.experiment.add_graph(self.discriminator,sampleImg)

        # ---- Log some of these images
        #
        z = torch.randn(self.hparams.batch_size, self.hparams.latent_dim)
        z = z.type_as(self.generator.model[0].weight)
        sample_imgs = self.generator(z)
        sample_imgs = sample_imgs.permute(0, 3, 1, 2) # from NHWC to NCHW
        grid = torchvision.utils.make_grid(tensor=sample_imgs, nrow=12, )
        self.logger.experiment.add_image(f"Generated images", grid,self.current_epoch)
