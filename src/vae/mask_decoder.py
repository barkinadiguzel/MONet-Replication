import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=128, image_size=64):
        super().__init__()
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_size * image_size)
        )

    def forward(self, z_slots):
        B, K, D = z_slots.shape

        masks = []
        for k in range(K):
            zk = z_slots[:, k]           
            logits = self.fc(zk)          
            logits = logits.view(B, 1, self.image_size, self.image_size)
            masks.append(logits)

        mask_logits = torch.cat(masks, dim=1)  

        return mask_logits
