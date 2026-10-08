# python module
import math
import os
import random
from contextlib import contextmanager

# torch and numpy imports
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tokenizers import Tokenizer

# megatron-core imports
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder
)

from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformer_from_scratch_01.config import (
    N_MODEL,
    S_MODEL,
    M_MODEL,
    L_MODEL,
    XL_MODEL,
)
# repo modules
from megatron_07.simple_tokenizer import SimpleTokenizer
from megatron_07.model_block import build_llama_model
from megatron_07.distributed_helper import wrap_model_for_distributed_training, build_optimizer
from megatron_07.dataset_helper import build_dataloader


#dataset
DATASET_FILEPATH = "megatron_07/megatron_2b"
TOKENIZER_PATH = "tokenizer_02/fineweb_bpe_32k.json"
CACHE_PATH = "megatron_07/cache"

MICRO_BATCH_SIZE = 4
NUM_MICROBATCHES = 4
CONTEXT_LENGTH = 512
VOCAB_SIZE = 32000
NUM_ITERATIONS = 100

MODEL_SIZE = "n"

# Parallelism
TP_SIZE = 2
PP_SIZE = 1
WORLD_SIZE = 2
assert WORLD_SIZE % (TP_SIZE * PP_SIZE) == 0
DP_SIZE = WORLD_SIZE // (TP_SIZE * PP_SIZE)

# Optimizer
LR = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 10
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

SEED = 1337

MASTER_ADDR = "localhost"
MASTER_PORT = "29500"

MODEL_SIZES = {
    "n": N_MODEL,
    "s": S_MODEL,
    "m": M_MODEL,
    "l": L_MODEL,
    "xl": XL_MODEL,
}
_CAUSAL_MASK = None

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

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=PP_SIZE,
    )


def cleanup():
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def get_lr(it):
    """Linear warmup + cosine decay."""
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    progress = (it - WARMUP_ITERS) / max(1, NUM_ITERATIONS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def get_causal_mask(device):
    global _CAUSAL_MASK
    if _CAUSAL_MASK is None:
        # True = masked out
        _CAUSAL_MASK = torch.triu(
            torch.ones(1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH,
                       dtype=torch.bool, device=device),
            diagonal=1,
        )
    return _CAUSAL_MASK


def forward_step_func(data_iterator, model):
    batch = next(data_iterator)

    # GPTDataset returns a dict.
    with time_block("Loading Batch")
        x = batch["tokens"].cuda(non_blocking=True).long()
        y = batch["labels"].cuda(non_blocking=True).long()
    loss_mask = batch["loss_mask"].cuda(non_blocking=True).float()

    batch_size = x.shape[0]
    position_ids = (
        torch.arange(CONTEXT_LENGTH, device=x.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    attention_mask = get_causal_mask(x.device)

    

    with time_block("Forward pass"):
        output = model(
            input_ids=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=y,
        )

    def loss_func(output_tensor):
        losses = output_tensor.float()
        loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return loss, {"lm loss": loss.detach()}

    return output, loss_func


def train(rank, world_size):
    setup(rank, world_size)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # Must run before model init so TP-region RNG states are set up.
    model_parallel_cuda_manual_seed(SEED)

    model = build_llama_model(MODEL_SIZES[MODEL_SIZE], VOCAB_SIZE, TP_SIZE, PP_SIZE, CONTEXT_LENGTH)
    model.cuda()
    model = wrap_model_for_distributed_training(model)

    optimizer = build_optimizer(model, LR, MIN_LR, WEIGHT_DECAY, GRAD_CLIP)

    train_dataloader = build_dataloader()
    train_iterator = iter(train_dataloader)

    forward_backward_func = get_forward_backward_func()

    for iteration in range(NUM_ITERATIONS):
        print(f"EPOCH:{iteration}")
        lr = get_lr(iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Clear BOTH Megatron-DDP's grad buffers and the optimizer's grads.
        model.zero_grad_buffer()
        optimizer.zero_grad()

        losses_per_microbatch = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=[model],                      # must be a list
            num_microbatches=NUM_MICROBATCHES,
            seq_length=CONTEXT_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            decoder_seq_length=CONTEXT_LENGTH,
            forward_only=False,
        )
        with time_block("Calculating Gradients"):
            # DP grad reduction + replicated-param (e.g. norm) grad sync.
            finalize_model_grads([model])

        with time_block("Backward pass"):
            update_successful, grad_norm, num_zeros = optimizer.step()

        with time_block("Loss"):
            # Average loss over microbatches, then over the DP group.
            mean_loss = torch.stack([l["lm loss"] for l in losses_per_microbatch]).mean()
            print(f"Loss: {mean_loss:0.5f}")

        with time_block("Synchronize"):
            dist.all_reduce(
                mean_loss,
                op=dist.ReduceOp.AVG,
                group=parallel_state.get_data_parallel_group(),
            )

        if rank == 0:
            print(
                f"iteration {iteration:4d} | lr {lr:.2e} | "
                f"loss {mean_loss.item():.4f} | "
                f"grad_norm {grad_norm} | "
                f"step_ok {update_successful}",
                flush=True,
            )

    cleanup()


if __name__ == "__main__":
    mp.spawn(
        train,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )