import torch
import torch.nn.functional as F

class RecurrentAttention:
    def __init__(self, attention_net, slots):
        self.attn = attention_net
        self.slots = slots

    def forward(self, x):
        B, C, H, W = x.shape
        scope = torch.ones(B, 1, H, W, device=x.device)
        masks = []

        for _ in range(self.slots):
            logits = self.attn(x, scope)
            mask = torch.sigmoid(logits) * scope
            masks.append(mask)
            scope = scope * (1 - mask)

        return masks
