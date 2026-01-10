import torch

def compose(recons, masks):
    out = torch.zeros_like(recons[0])
    for xk, mk in zip(recons, masks):
        out += xk * mk
    return out
