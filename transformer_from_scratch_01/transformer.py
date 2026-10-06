from transformer_from_scratch_01.building_blocks import Embedding, TransformerBlock, RMSNorm, LinearProjection
from transformer_from_scratch_01.config import N_MODEL, S_MODEL, M_MODEL, L_MODEL, XL_MODEL
import torch
from torch import nn



class TransformerLM(nn.Module):
  def __init__(
      self,
      vocab_size: int,
      context_length: int,
      theta: float = 10000.,
      mask: torch.Tensor | None = None,
      model_size: str = 'm'):

    super().__init__()

    if model_size == 'n':
    	(d_model, d_ff, num_layers, num_heads) = (value for k, value in N_MODEL.items())
    if model_size == 's':
    	(d_model, d_ff, num_layers, num_heads) = (value for k, value in S_MODEL.items())
    if model_size == 'm':
    	(d_model, d_ff, num_layers, num_heads) = (value for k, value in M_MODEL.items())
    if model_size == 'l':
    	(d_model, d_ff, num_layers, num_heads) = (value for k, value in L_MODEL.items())
    if model_size == 'xl':
    	(d_model, d_ff, num_layers, num_heads) = (value for k, value in XL_MODEL.items())


    self.token_embedding = Embedding(vocab_size, d_model)

    self.layers = nn.ModuleList([
        TransformerBlock(
            d_model = d_model,
            num_heads = num_heads,
            context_length = context_length,
            d_ff = d_ff,
            theta = theta,
            mask = mask
        ) for _ in range(num_layers)
    ])

    self.final_norm = RMSNorm(d_model)
    self.lm_head = LinearProjection(d_model, vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    x = self.token_embedding(x)

    for layer in self.layers:
      x = layer(x)

    x = self.final_norm(x)

    logits = self.lm_head(x)

    return logits