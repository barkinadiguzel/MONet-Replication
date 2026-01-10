import torch.nn as nn

class AttentionNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, scope):
        inp = torch.cat([x, scope], dim=1)
        logits = self.net(inp)
        return logits
