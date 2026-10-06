#batch passing the gradient
import torch
import torch.distributed as dist
import os
import torch.nn.functional as F
import torch.multiprocessing as mp
from naive_bpe_dataset import MemmapTokenDataset

import importlib
# Programmatically import the module using its string name
transformer_mod = importlib.import_module("01_transformer") 


CONTEXT_LENGTH = 1024
VOCAB_SIZE = 10001
MODEL_SIZE = "s"
BATCH = 16

# Optimizer
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.1

# Epochs
EPOCH = 100

# No. of GPU or world_size
WORLD_SIZE = 2

# distribution
MASTER_ADDR = "localhost"
MASTER_PORT = 29500
BACKEND = "nccl"

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = MASTER_ADDR
  os.environ['MASTER_PORT'] = MASTER_PORT

  dist.init_process_group(backend=BACKEND,rank = rank, world_size = WORLD_SIZE)

def distributed(rank, world):
  setup(rank, world_size)

  device = torch.device(f"cuda:{rank}")

  model = transformer_mod.TransformerLM(
      context_length = CONTEXT_LENGTH,
      vocab_size = VOCAB_SIZE,
      model_size = 's'
      ).to(device)

  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=LR,
      betas=(BETA1, BETA2),
      eps=EPS,
      weight_decay=WEIGHT_DECAY
      )

  train_dataset = MemmapTokenDataset(
      file_path = "TinyStories-train.bin",
      seq_len = CONTEXT_LENGTH,
      split="train",
      train_fraction=0.99
      )

  val_dataset = MemmapTokenDataset(
      file_path = "TinyStories-train.bin",
      seq_len = CONTEXT_LENGTH,
      split="val",
      train_fraction=0.99,
      )

  for params in model.parameters():
    dist.broadcast(params.data, src=0)


  for i in range(epoch):

    x, y = train_dataset.get_batch(BATCH//WORLD_SIZE)
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat.reshape(-1, y_hat.size(-1)),y.reshape(-1))
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    params = [param for param in model.parameters() if param.grad is not None]

    flattened_grad = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flattened_grad, op=dist.ReduceOp.SUM)
    flattened_grad /= world_size

    unflat_grads = torch._utils._unflatten_dense_tensors(flattened_grad, grads)

    for param, grad in zip(params, unflat_grads):
        param.grad.copy_(grad)

    optimizer.step()

    if rank == 0:
      print(f"step={i:04d}, loss={loss.item():.4f}",flush=True)

  dist.destroy_process_group()


if __name__ == "__main__":
  mp.spawn(
      distributed,
      args=(world_size,),
      nprocs=world_size,
      join=True)


