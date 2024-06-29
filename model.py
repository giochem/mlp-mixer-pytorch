import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange

class MLPMixer(nn.Module):
    def __init__(self, width=300, height=300, c=3, num_classes=10, patch_size=100, num_blocks=2, hidden_dim=512, tokens_mlp_dim=2048, channels_mlp_dim=256):
        super(MLPMixer, self).__init__()
        self.c = c
        self.s = int((width * height) / (patch_size * patch_size))
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.tokens_nlp_dim = tokens_mlp_dim
        self.channels_nlp_dim = channels_mlp_dim

        self.per_patch = nn.Linear(self.c * self.patch_size * self.patch_size, self.hidden_dim)
        self.token_mlp_block = nn.Sequential(
            nn.Linear(self.s, self.tokens_nlp_dim),
            nn.GELU(),
            nn.Linear(self.tokens_nlp_dim, self.s),
        )
        self.channel_mlp_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.channels_nlp_dim),
            nn.GELU(),
            nn.Linear(self.channels_nlp_dim, self.hidden_dim),
        )
        self.output = nn.Linear(self.hidden_dim, self.num_classes)

    def mixer_block(self, x):
      y = nn.LayerNorm(x.shape)(x)
      y = torch.transpose(y, 1, 2)
      # MlpBlock
      y = self.token_mlp_block(y)

      y = torch.transpose(y, 1, 2)
      # skipping connection
      x = x + y
      y = nn.LayerNorm(x.shape)(x)
      return x + self.channel_mlp_block(y)

    def forward(self, x):
      p = self.patch_size
      x = x.unfold(2, p, p).unfold(3, p, p)
      x = rearrange(x, "b c s1 s2 h w -> b (s1 s2) (h w c)")

      x = self.per_patch(x)

      for _ in range(self.num_blocks):
        x = self.mixer_block(x)
      x = torch.mean(x, 1)

      return self.output(x)

# BATCH C H W
# h, w = 32, 32
# model = NeuralNetwork(width=w, height=h, num_classes=2, patch_size=4)
# summary(model, input_size=(16, 3, h, w))