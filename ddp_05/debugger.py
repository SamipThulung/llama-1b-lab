# nccl_test.py on 2 gpu
import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp

def run(rank, ws):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    x = torch.ones(128_000_000, device=f"cuda:{rank}")  # ~512 MB
    for i in range(3):
        torch.cuda.synchronize(); t = time.perf_counter()
        dist.broadcast(x, src=0)
        torch.cuda.synchronize()
        if rank == 0: print(f"broadcast {i}: {time.perf_counter()-t:.3f}s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)