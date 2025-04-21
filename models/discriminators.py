"""
# Dans la class Cycle GAN que nous allons créeer on doit créer
les méthodes suivantes pour appliquer les dissriminateurs sur les
les générateurs Ga et Gb.

Module: discriminators.py

Defines standalone PatchGAN discriminators (D_X and D_Y) for CycleGAN-style unpaired image translation.
Usage:
    from discriminators import define_D

    # Instantiate discriminator for domain X (e.g., 3-channel RGB)
    D_X = define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3,
                   norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[0])

    # Instantiate discriminator for domain Y (e.g., 3-channel RGB)
    D_Y = define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3,
                   norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
"""
import torch
import torch.nn as nn

def weights_init_normal(m, mean=0.0, std=0.02):
    """
    Initialize convolutional and normalization layers with a normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean, std)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm2d') != -1 or classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0.0)

class NLayerDiscriminator(nn.Module):
    """
    A 70×70 PatchGAN Discriminator.

    Architecture:
      1. Conv (no norm) + LeakyReLU
      2. (n_layers-1) × [Conv + Norm + LeakyReLU]
      3. Conv + Norm + LeakyReLU (stride=1)
      4. Final Conv to 1-channel output
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw, padw = 4, 1
        layers = []
        # 1. Initial conv block
        layers += [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        # 2. Hidden layers with increasing filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_prev, nf_mult = nf_mult, min(2**n, 8)
            layers += [
                nn.Conv2d(ndf*nf_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # 3. Penultimate layer (stride=1)
        nf_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf*nf_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf*nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 4. Final 1-channel conv
        layers += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def define_D(input_nc, ndf=64, netD='basic', n_layers_D=3,
             norm='instance', init_type='normal', init_gain=0.02, gpu_ids=None):
    """
    Instantiate a PatchGAN discriminator for domain X or Y.

    Args:
        input_nc (int):      # channels of input images
        ndf (int):           # filters in first layer
        netD (str):          # only 'basic' supported currently
        n_layers_D (int):    # number of downsampling layers
        norm (str):          # 'instance' or 'batch'
        init_type (str):     # 'normal'
        init_gain (float):   # weight init std
        gpu_ids (list):      # optional list of GPU device ids

    Returns:
        nn.Module: PatchGAN discriminator
    """
    # Select normalization
    if norm == 'instance': norm_layer = nn.InstanceNorm2d
    elif norm == 'batch': norm_layer = nn.BatchNorm2d
    else: raise ValueError(f"Unsupported norm: {norm}")

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer)
    else:
        raise NotImplementedError(f"Discriminator {netD} not implemented")

    # Initialize weights
    net.apply(lambda m: weights_init_normal(m, mean=0.0, std=init_gain))

    # Move to GPU and wrap in DataParallel
    if gpu_ids and torch.cuda.is_available():
        net.to(f'cuda:{gpu_ids[0]}')
        net = nn.DataParallel(net, gpu_ids)

    return net

# Example usage/

if __name__ == "__main__":

    # Pour domaine A
    D_A = define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3,
                norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[0])

    # Pour domaine B
    D_B = define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3,
                norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
