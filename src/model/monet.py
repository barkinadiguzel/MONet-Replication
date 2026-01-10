import torch
import torch.nn as nn

from attention.recurrent_attention import RecurrentAttention
from vae.component_vae import ComponentVAE
from decoder.compositor import compose


class MONet(nn.Module):
    def __init__(self, attention_net, component_vae, mask_decoder, slots):
        super().__init__()

        self.attention = RecurrentAttention(attention_net, slots)
        self.component_vae = component_vae
        self.mask_decoder = mask_decoder
        self.slots = slots

    def forward(self, x):
        # 1) Recursive attention → masks
        masks = self.attention.forward(x)   

        recons = []
        zs = []
        mus = []
        logvars = []

        # 2) Component-wise VAE
        for mk in masks:
            recon, z, mu, logvar = self.component_vae.forward(x, mk)
            recons.append(recon)
            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        # (B, K, D)
        z_slots = torch.stack(zs, dim=1)

        # 3) Decode masks from latents
        mask_logits = self.mask_decoder(z_slots)

        # 4) Compose final image
        x_recon = compose(recons, masks)

        return {
            "x_recon": x_recon,
            "recons": recons,
            "masks": masks,
            "z_slots": z_slots,
            "mus": mus,
            "logvars": logvars,
            "mask_logits": mask_logits
        }
