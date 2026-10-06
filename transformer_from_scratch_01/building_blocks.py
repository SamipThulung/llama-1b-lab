#1. LLM building blocks
import math
import torch
import torch.nn as nn
from einops import rearrange, einsum

class LinearProjection(nn.Module):
  def __init__(self, input_feature: int, output_feature: int):

    super().__init__()
    self.weights = nn.Parameter(torch.empty(input_feature, output_feature))

    std = math.sqrt(2/(input_feature + output_feature))

    nn.init.trunc_normal_(
        self.weights,
        mean = 0,
        std = std,
        a = -3 * std,
        b = 3 * std
    )

  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    return einsum(
        input_tensor,
        self.weights,
        'batch seq input_features, input_features seq_out -> batch seq seq_out')

class Embedding(nn.Module):
  def __init__(self, vocab_size: int, emb_dim: int):

    super().__init__()
    self.weights = nn.Parameter(
        torch.empty((vocab_size, emb_dim))
    )

    nn.init.trunc_normal_(
        self.weights,
        mean = 0,
        std = 1,
        a = -3,
        b = 3
    )

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens = (batch, sequence)
    output = (batch, sequence, emb_dim)
    """
    return self.weights[tokens]

class RMSNorm(nn.Module):
  def __init__(self, d_model, eps = 1e-5):
    super().__init__()

    self.eps = eps

    self.weights = nn.Parameter(
        torch.ones(d_model)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    # works on the last dimension
    x_mean = torch.mean(x ** 2, dim = -1, keepdim=True)

    x_rsqrt = torch.rsqrt(x_mean + self.eps)

    x_norm = x * x_rsqrt * self.weights

    return x_norm

class FFP(nn.Module):
  def __init__(self, d_model:int, d_ff: int):
    super().__init__()



    self.W1 = LinearProjection(d_model, d_ff)
    self.W3 = LinearProjection(d_model, d_ff)

    self.W2 = LinearProjection(d_ff, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    x_w1 = self.W1(x)
    x_w3 = self.W3(x)

    x_w1 = x_w1 * torch.sigmoid(x_w1)

    x_w3 = x_w1 * x_w3

    return self.W2(x_w3)

class ROPE(nn.Module):
  def __init__(self, d_k: int, theta: float, context_length: int):
    super().__init__()

    self.d_k = d_k

    self.inv_freq = 1/ (theta ** (torch.arange(0, self.d_k, 2)/ self.d_k))
    self.t = torch.arange(0, context_length)
    self.freq = einsum(self.t, self.inv_freq, 'context_length, half_dk -> context_length half_dk')

    self.emb = torch.concat([self.freq, self.freq], dim = -1)

    self.register_buffer("cos_cached", self.emb.cos(), persistent = False)
    self.register_buffer("sin_cached", self.emb.sin(), persistent = False)

  def _rotate_half(self, x):
    x1 = x[..., :self.d_k // 2 ]
    x2 = x[..., self.d_k // 2: ]

    return torch.cat([-x2, x1], dim = -1)

  def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    cos = self.cos_cached[tokens]
    sin = self.sin_cached[tokens]
    return (x * cos) + (self._rotate_half(x) * sin)

def softmax(x: torch.Tensor, dim: int):

  max_val = torch.amax(x, dim=dim, keepdim=True)
  stable_x = x - max_val

  exp_x = torch.exp(stable_x)

  sum_x = torch.sum(exp_x, dim = dim, keepdim = True)

  return exp_x / sum_x


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None) -> torch.Tensor:

  d_k = q.shape[-1]
  qk_norm = einsum(
      q, k,
      '... q_seq dim, ... k_seq dim -> ... q_seq k_seq'
      ) / math.sqrt(d_k)



  if mask is not None:
    qk_norm = qk_norm.masked_fill(~mask, float("-inf"))

  qk_softmax = softmax(qk_norm, dim=-1)

  output = einsum(
      qk_softmax, v,
      '... q_seq k_dim, ... k_dim v_dim -> ... q_seq v_dim'
  )
  return output

class MultiHeadAttention(nn.Module):
  def __init__(
      self,
      d_model: int,
      num_head: int,
      context_length: int,
      mask: torch.Tensor | None = None, 
      theta: float = 10000.0
  ):
      super().__init__()
      self.d_model = d_model
      self.num_head = num_head
      self.head_dim = d_model // num_head

      self.mask = mask
      self.q_proj = LinearProjection(d_model, d_model)
      self.k_proj = LinearProjection(d_model, d_model)
      self.v_proj = LinearProjection(d_model, d_model)

      self.rope = ROPE(
          d_k = self.head_dim,
          theta = theta,
          context_length = context_length
      )
      self.outputW = LinearProjection(d_model, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

      batch_size, context_length, _ = x.shape

      q = self.q_proj(x)
      k = self.k_proj(x)
      v = self.v_proj(x)

      q = rearrange(
          q,
          'batch context_length (num_head d_k) -> batch num_head context_length d_k', num_head = self.num_head)
      k = rearrange(
          k,
          'batch context_length (num_head d_k) -> batch num_head context_length d_k', num_head = self.num_head)
      v = rearrange(
          v,
          'batch context_length (num_head d_k) -> batch num_head context_length d_k', num_head = self.num_head)

      positions = torch.arange(context_length, device = x.device)

      q = self.rope(q, positions)
      k = self.rope(k, positions)

      attention_output = scaled_dot_product(
          q,
          k,
          v,
          self.mask
      )

      attention_output = rearrange(
          attention_output,
          'batch num_head seq dk -> batch seq (num_head dk)'
          )

      return self.outputW(attention_output)



class TransformerBlock(nn.Module):
  def __init__(self, d_model, num_heads, context_length, mask, d_ff, theta):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.context_length = context_length
    self.d_ff = d_ff

    self.rmsnorm_mha = RMSNorm(d_model)
    self.mha = MultiHeadAttention(d_model, num_heads, context_length, mask, theta)

    self.rmsnorm_ff = RMSNorm(d_model)
    self.ff = FFP(d_model, d_ff)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    x_mha = self.mha(self.rmsnorm_mha(x))
    x = x + x_mha

    x_ff = self.ff(self.rmsnorm_ff(x))
    x = x + x_ff

    return x





