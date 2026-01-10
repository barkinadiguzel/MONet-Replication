import torch
import torch.nn.functional as F


class MONetLoss:
    def __init__(self, beta=1.0, gamma=1.0):
        self.beta = beta
        self.gamma = gamma

    def reconstruction_loss(self, x, recons, masks):
        loss = 0.0
        for xk, mk in zip(recons, masks):
            loss += ((x - xk) ** 2 * mk).mean()
        return loss

    def kl_latent(self, mus, logvars):
        kl = 0.0
        for mu, logvar in zip(mus, logvars):
            kl += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def kl_masks(self, attn_masks, mask_logits):
        # q_psi(c|x)
        q = torch.cat(attn_masks, dim=1)  # (B, K, H, W)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)

        # p_theta(c|z)
        p = F.softmax(mask_logits, dim=1)

        kl = (q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))).mean()
        return kl

    def __call__(self, x, outputs):
        recons = outputs["recons"]
        masks = outputs["masks"]
        mus = outputs["mus"]
        logvars = outputs["logvars"]
        mask_logits = outputs["mask_logits"]

        recon_loss = self.reconstruction_loss(x, recons, masks)
        kl_z = self.kl_latent(mus, logvars)
        kl_masks = self.kl_masks(masks, mask_logits)

        total_loss = recon_loss + self.beta * kl_z + self.gamma * kl_masks

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl_z": kl_z,
            "kl_masks": kl_masks
        }
