import torch
import torch.distributed as dist
from fsdp_06.layer_shard import ShardedLinear, ShardedEmbedding
from torch import nn
import weakref



class FSDP(nn.Module):
    def __init__(self, module: nn.Module, _get_comm_stream, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._layers = []
        self.comm_stream = _get_comm_stream()

        self._sync_initial_parameters()
        self._wrap_module(self.module)

        sharded_ids = set()
        for i, layer in enumerate(self._layers):
            layer._owner = weakref.ref(self)
            layer._index = i
            layer.comm_stream = self.comm_stream
            for p in layer.parameters():
                sharded_ids.add(id(p))

        self._replicated_parameters = [
            p for p in self.module.parameters() if id(p) not in sharded_ids
        ]

    def _sync_initial_parameters(self):
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=0)

    def _wrap_module(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                wrapped = ShardedLinear(child, self.rank, self.world_size, self.compute_dtype)
                setattr(module, name, wrapped)
                self._layers.append(wrapped)
            elif isinstance(child, nn.Embedding):
                wrapped = ShardedEmbedding(child, self.rank, self.world_size, self.compute_dtype)
                setattr(module, name, wrapped)
                self._layers.append(wrapped)
            else:
                self._wrap_module(child)

    def forward(self, *inputs, **kwargs):
        # Bootstrap the pipeline: layers 0 and 1; afterwards each layer's
        # forward triggers the gather of layer (index + 2).
        self._prefetch(0)
        self._prefetch(1)
        if self.compute_dtype is not None:
            with torch.autocast("cuda", dtype=self.compute_dtype):
                return self.module(*inputs, **kwargs)
        return self.module(*inputs, **kwargs)
    def finish_gradient_synchronization(self):
        for layer in self._layers:
            layer._grad_sync()
            layer._release_gather_weights()  # drop any leftover prefetched weights

        for p in self._replicated_parameters:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.world_size)

        torch.cuda.current_stream().wait_stream(self.comm_stream)

    def _prefetch(self, index):
        if index < 0 or index >= len(self._layers):
            return
        self._layers[index]._start_gather()


