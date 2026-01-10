import torch
import torch.nn as nn

class ComponentVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        z, mu, logvar = self.encoder(x, mask)
        recon = self.decoder(z)
        return recon, z, mu, logvar
