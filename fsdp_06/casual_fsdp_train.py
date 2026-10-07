import os 
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp
    
from fsdp_06.fsdp_wrapper import FSDP
from transformer_from_scratch_01.toy_model import TinyModel
import random
import numpy as np
import time

from contextlib import contextmanager
import torch.nn.functional as F

BATCH_SIZE = 16
CONTEXT_LENGTH = 512
VOCAB_SIZE = 10001
EPOCH = 100

MASTER_ADDR = "localhost"
MASTER_PORT = "29500"

WORLD_SIZE = 2
VERBOSE_TIMING = True

_COMM_STREAM = None

def _get_comm_stream():
    global _COMM_STREAM
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream()
    return _COMM_STREAM


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
    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD_SIZE, device_id=torch.device(f"cuda:{rank}"))


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, losses):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    seed = 1337 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with time_block("Model loaded"):
        model = TinyModel().to(device)

    with time_block("Model wrapped"):
        model = FSDP(model, _get_comm_stream, compute_dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(total_params, f"{total_params / 1e6:.2f}M", flush=True)

    with time_block("Initializing x and y"):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)

    with time_block("Warmup"):
        for _ in range(5):
            model(x)
        
    for i in range(EPOCH):
    
        with time_block(f"Forward pass rank: {rank}"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), targets.reshape(-1))
        
        with time_block(f"Backward pass rank: {rank}"):
            loss.backward()
        
        print(f"[Rank{rank}] Epoch{i}: loss = {loss.item():.4f}")

        with time_block(f"Gradient Synchronize"):
            model.finish_gradient_synchronization()

        with time_block(f"Updating parameters"):
            optimizer.step()
            
        loss_val = loss.item()
        if rank == 0:
            losses.put((i, loss_val))
            print(f"Epoch{i}: loss = {loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    # WORLD_SIZE = torch.cuda.device_count()

    ctx = mp.get_context("spawn")
    loss_queue = ctx.Queue()

    mp.spawn(
        train,
        args=(WORLD_SIZE, loss_queue),
        nprocs=WORLD_SIZE,
        join=True,
    )

    while not loss_queue.empty():
        i, loss = loss_queue.get()
        print(f"Step: {i}, loss: {loss:.5f}")

