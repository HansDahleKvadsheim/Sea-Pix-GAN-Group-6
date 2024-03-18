import torch as th
import torch.nn as nn
import torch.nn.functional as F

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device {device}")

#===============================================================================
# Discriminator architecture ===================================================
#===============================================================================

class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(leaky_relu_slope)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x      
    
class ZeroPadModule(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_slope=0.2):
        super().__init__()
        
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        return x

class Discriminator(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.DownLayers = nn.Sequential(
            DownModule(6, 64),
            DownModule(64, 128),
            DownModule(128, 256),
            ZeroPadModule(256, 256),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ZeroPadModule(512, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )
        
    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        """Forward pass of the discriminator

        Args:
            x (th.Tensor): Raw underwater image
            y (th.Tensor): Enhanced underwater image

        Returns:
            th.Tensor: Output tensor measuring the realness of the input images
        """
        
        z = th.concatenate((x, y), dim=1)
        
        # Input tensor shape
        print("Input tensor shape:")
        print(y.shape)
        
        # TODO: Convolutions
        
        for layer in self.DownLayers:
            z = layer(z)
            print(z.shape)
        
        return z
    
discriminator = Discriminator().to(device)

sample = th.randn(1, 3, 256, 256, device=device)
clone = sample.clone()
output = discriminator(sample, clone)

#===============================================================================
# Generator architecture =======================================================
#===============================================================================

class EncoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(leaky_relu_slope)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x      

class FeatureMapModule(nn.Module):
        def __init__(self, in_channels, out_channels, leaky_relu_slope=0.2):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.lrelu = nn.LeakyReLU(leaky_relu_slope)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.lrelu(x)
            return x      

class DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
    
class OutputModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            
        def forward(self, x):
            x = self.deconv(x)
            return x

class Autoencoder(nn.Module):
    """
    Autoencoder model for image generation

    A residual autoencoder model for image generation. 
    The final model will be an image-to-image translation model
    that enhances underwater images.
    """
    def __init__(self):
        super().__init__()

        self.EncoderLayers = nn.ModuleList([
            EncoderModule(3, 64),
            EncoderModule(64, 128),
            EncoderModule(128, 256),
            EncoderModule(256, 512),
            EncoderModule(512, 512),
            EncoderModule(512, 512),
            EncoderModule(512, 512),
            FeatureMapModule(512, 512),
        ])
        
        self.DecoderLayers = nn.ModuleList([
            DecoderModule(1024, 512),
            DecoderModule(1024, 512),
            DecoderModule(1024, 512),
            DecoderModule(1024, 512, dropout_prob=0.0),
            DecoderModule(1024, 256, dropout_prob=0.0),
            DecoderModule(512, 128, dropout_prob=0.0),
            DecoderModule(256, 64, dropout_prob=0.0),
        ])
        
        self.OutputLayer = OutputModule(128, 3)
        
    def forward(self, x, z):
        """Forward pass for the autoencoder model.

        Args:
            x (th.Tensor): Input image tensor
            z (th.Tensor): Noise tensor

        Returns:
            th.Tensor: Output image tensor
        """

        #TODO: Figure out precisely how the noise tensor is used. Tentaively we add them together.
        x = x + z

        # Store the activations of the encoder layers for skip connections
        layer_outputs = []
        
        print("Starting forward pass")
        print(x.shape)
        
        # Encoder pass
        for i in range(len(self.EncoderLayers)):
            x = self.EncoderLayers[i](x)
            layer_outputs.append(x)
            print(x.shape)
            
        print("Encoding complete")
        print(x.shape)
        
        
        # Decoder pass
        #TODO: Verify that that the first layer of decoding is correct
        
        for i in range(len(self.DecoderLayers)):
            
            # Get the appropriate encoder activation
            s = layer_outputs.pop()
            
            # If the shapes match, concatenate the activations
            if x.shape == s.shape:
                x = th.cat((x, s), 1)
                
            else:
                print("Error, shapes do not match")
                return th.tensor([])

            # Pass the concatenated activations through the decoder layer
            x = self.DecoderLayers[i](x)
            print(x.shape)
                          
        print("Decoding complete")
        
        # Perform the final deconvolution
        x = th.cat((x, layer_outputs.pop()), 1)
        x = self.OutputLayer(x)
        print(x.shape)
        print("Output complete")
            
        return x
 
generator = Autoencoder().to(device)

sample = th.randn(1, 3, 256, 256, device=device)
noise = th.randn(1, 3, 256, 256, device=device)
output = generator(sample, noise)