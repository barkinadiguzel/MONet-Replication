import torch
import torch.nn.functional as F

class MaskGenerator:
    def __init__(self, attention_net):
        self.attn = attention_net

    def generate(self, x, scope):
        logits = self.attn(x, scope)
        mask = torch.sigmoid(logits) * scope
        return mask
