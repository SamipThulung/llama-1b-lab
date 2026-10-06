#batch passing the gradient
import torch
import torch.distributed as dist
import os
import torch.nn.functional as F
import torch.multiprocessing as mp
from naive_bpe_dataset import MemmapTokenDataset
from transformer_from_scratch_01.transformer import TransformerLM
import time
from contextlib import contextmanager


CONTEXT_LENGTH = 512
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
MASTER_PORT = "29500"
BACKEND = "nccl"

@contextmanager
def time_block(name="Code block"):
    start = time.perf_counter()
    try:
        yield  
    finally:
        end = time.perf_counter()
        print(f"[{name}] took {end - start:.4f} seconds")

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = MASTER_ADDR
  os.environ['MASTER_PORT'] = MASTER_PORT

  dist.init_process_group(backend=BACKEND,rank = rank, world_size = WORLD_SIZE)

def distributed(rank, world_size):
  setup(rank, world_size)

  device = torch.device(f"cuda:{rank}")

  with time_block("Load model and Optimizer"):
    model = TransformerLM(
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

  with time_block("Warm up"):
    x = torch.randint(low=0, high=10001, size=(16, 512))
    for j in range(0,5):
      model(x)
    del x


  with time_block("Load dataset"):
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

  with time_block("Broadcast model"):
    for params in model.parameters():
      dist.broadcast(params.data, src=0)
    torch.cuda.synchronize(device)

  print("Training")
  for i in range(epoch):

    with time_block("Getting batch and loading to device"):
      x, y = train_dataset.get_batch(BATCH//WORLD_SIZE)
      x = x.to(device)
      y = y.to(device)

    optimizer.zero_grad()
    with time_block("forward pass"):
      y_hat = model(x)
      loss = F.cross_entropy(y_hat.reshape(-1, y_hat.size(-1)),y.reshape(-1))
      losses.put((i, loss.detach().cpu().item()))
    print("*"*9)
    print(f"Loss: {loss}")
    print("*"*9)
    print()

    with time_block("Calculating Gradient"):
      loss.backward()

    with time_block("Preparing data for reduce scatter"):
      grads = [param.grad for param in model.parameters() if param.grad is not None]
      params = [param for param in model.parameters() if param.grad is not None]

      flattened_grad = torch._utils._flatten_dense_tensors(grads)

    with time_block("All reduce Operation"):
      dist.all_reduce(flattened_grad, op=dist.ReduceOp.SUM)

    with time_block("Post gather normalization and unflattening")
      flattened_grad /= world_size
      unflat_grads = torch._utils._unflatten_dense_tensors(flattened_grad, grads)

    with time_block(f"Copying Grad to the current rank {rank}")
      for param, grad in zip(params, unflat_grads):
          param.grad.copy_(grad)

    with time_block("Backward pass"):
      optimizer.step()

    if rank == 0:
      print(f"step={i:04d}, loss={loss.item():.4f}",flush=True)

  dist.destroy_process_group()


if __name__ == "__main__":
  losses = []

  ctx = mp.get_context('spawn')
  loss_queue = ctx.Queue()

  mp.spawn(
      distributed,
      args=(world_size,loss_queue),
      nprocs=world_size,
      join=True)


