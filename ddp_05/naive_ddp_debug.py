# Manual DDP: flattened gradient all-reduce, with honest (synchronized) timing
import os
import time
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from ddp_05.naive_dataset import MemmapTokenDataset
from transformer_from_scratch_01.transformer import TransformerLM


CONTEXT_LENGTH = 512
VOCAB_SIZE = 10001
MODEL_SIZE = "l"
BATCH = 16            # global batch; each rank uses BATCH // WORLD_SIZE
USE_BF16 = True       # bf16 autocast (A40 supports it)

# Optimizer
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.1

# Steps
EPOCH = 100

# No. of GPUs or world_size
WORLD_SIZE = 2

# Distribution
MASTER_ADDR = "localhost"
MASTER_PORT = "29500"
BACKEND = "nccl"

FILEPATH = "dataset_03/TinyStories-train.bin"

VERBOSE_TIMING = True  # only rank 0 prints timings


@contextmanager
def time_block(name="Code block", rank=0):
    # CUDA is async: synchronize before and after so the timing is honest
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        if VERBOSE_TIMING and rank == 0:
            print(f"[{name}] took {time.perf_counter() - start:.4f} seconds", flush=True)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))


def distributed(rank, world_size, losses):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Different data order per rank
    seed = 1337 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mask = torch.tril(
        torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH, dtype=torch.bool, device=device)
    )

    with time_block("Load model and optimizer", rank):
        model = TransformerLM(
            context_length=CONTEXT_LENGTH,
            vocab_size=VOCAB_SIZE,
            model_size=MODEL_SIZE,
            mask=mask,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=(BETA1, BETA2),
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(total_params, f"{total_params / 1e6:.2f}M", flush=True)

    # Warm-up: CUDA/cuBLAS init, lazy kernel loading. Kept separate from comm timing.
    with time_block("Warm up", rank):
        x = torch.randint(0, VOCAB_SIZE, (BATCH // world_size, CONTEXT_LENGTH), device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            for _ in range(5):
                model(x)
        del x

    # Initialize NCCL communicators here so the cost is NOT counted in the broadcast
    with time_block("NCCL init (barrier)", rank):
        dist.barrier(device_ids=[rank])

    with time_block("Load dataset", rank):
        train_dataset = MemmapTokenDataset(
            file_path=FILEPATH,
            seq_len=CONTEXT_LENGTH,
            split="train",
            train_fraction=0.99,
        )
        val_dataset = MemmapTokenDataset(  # noqa: F841 (unused for now)
            file_path=FILEPATH,
            seq_len=CONTEXT_LENGTH,
            split="val",
            train_fraction=0.99,
        )

    # Single flattened broadcast instead of one call per parameter tensor
    with time_block("Broadcast model", rank):
        param_data = [p.data for p in model.parameters()]
        flat = torch._utils._flatten_dense_tensors(param_data)
        dist.broadcast(flat, src=0)
        for p, new in zip(model.parameters(),
                          torch._utils._unflatten_dense_tensors(flat, param_data)):
            p.data.copy_(new)
        del flat

    if rank == 0:
        print("Training", flush=True)

    for i in range(EPOCH):
        with time_block("Getting batch and loading to device", rank):
            x, y = train_dataset.get_batch(BATCH // world_size)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with time_block("Forward pass", rank):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                y_hat = model(x)
                loss = F.cross_entropy(
                    y_hat.reshape(-1, y_hat.size(-1)).float(), y.reshape(-1)
                )

        with time_block("Backward pass (compute gradients)", rank):
            loss.backward()

        with time_block("Flatten grads", rank):
            params = [p for p in model.parameters() if p.grad is not None]
            grads = [p.grad for p in params]
            flat_grad = torch._utils._flatten_dense_tensors(grads)

        with time_block("All-reduce", rank):
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)

        with time_block("Normalize, unflatten, copy back", rank):
            flat_grad /= world_size
            for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat_grad, grads)):
                p.grad.copy_(g)

        with time_block("Optimizer step", rank):
            optimizer.step()

        loss_val = loss.item()
        if rank == 0:
            losses.put((i, loss_val))
            print(f"step={i:04d}, loss={loss_val:.4f}", flush=True)

    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    loss_queue = ctx.Queue()

    mp.spawn(
        distributed,
        args=(WORLD_SIZE, loss_queue),
        nprocs=WORLD_SIZE,
        join=True,
    )

    while not loss_queue.empty():
        i, loss = loss_queue.get()
        print(f"Step: {i}, loss: {loss:.5f}")