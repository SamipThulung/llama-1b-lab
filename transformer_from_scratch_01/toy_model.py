# Tiny model
import torch
from torch import nn
import torch.nn.functional as F

class TinyModel(nn.Module):

    def __init__(self, vocab_size=10001, d_model=256):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, )
        self.linear1 = nn.Linear(256,1024,)
        self.norm1 = nn.LayerNorm(1024,)
        self.linear2 = nn.Linear(1024,256,)

        self.norm2 = nn.LayerNorm(256,)

        self.lm_head = nn.Linear(256,vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.lm_head(x)
        return x