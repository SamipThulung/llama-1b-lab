
import torch
from torch import nn
import math
import torch.distributed as dist

class SharedBase(nn.Module):
    """shard creation, all-gather, reduce-scatter, release."""

    def __init__(self, rank, world_size, compute_dtype):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.compute_dtype = compute_dtype  # may be None -> use param dtype

        self.comm_stream = _get_comm_stream()

        self.register_parameter("bias", None)
        self.original_bias_shape = 0
        self.shard_bias_size = 0

        self._full_weights = None
        self._full_bias = None
        self._gather_event = None
        self._pending_grad_work = []
        self._owner = None
        self._index = None

    def _dtype(self):
        return self.compute_dtype if self.compute_dtype is not None else self.weights.dtype

    def _create_shard(self, full_parameter):
        # Master shard stays in the ORIGINAL dtype (FP32).
        full_parameter = full_parameter.detach().contiguous()

        original_rows = full_parameter.shape[0]
        shard_size = math.ceil(original_rows / self.world_size)
        total_rows = self.world_size * shard_size

        if original_rows != total_rows:
            padded = torch.zeros(
                (total_rows, *full_parameter.shape[1:]),
                dtype=full_parameter.dtype,
                device=full_parameter.device,
            )
            padded[:original_rows].copy_(full_parameter)
        else:
            padded = full_parameter

        start = self.rank * shard_size
        local_shard = padded[start:start + shard_size].clone().contiguous()
        return nn.Parameter(local_shard), original_rows, shard_size

    def _gather_param(self, shard, original_rows, shard_rows):
        dtype = self._dtype()
        local = shard.detach().to(dtype).contiguous()  # cast BEFORE communicating
        full = torch.empty(
            (self.world_size * shard_rows, *local.shape[1:]),
            dtype=dtype,
            device=local.device,
        )
        dist.all_gather_into_tensor(full, local)
        return full[:original_rows]

    def _start_gather(self):
        if self._full_weights is not None:
            return self._full_weights, self._full_bias

        # Make sure the latest optimizer update to the shard is visible.
        self.comm_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.comm_stream):
            self._full_weights = self._gather_param(
                self.weights, self.original_shape, self.shard_size
            )
            if self.bias is not None:
                self._full_bias = self._gather_param(
                    self.bias, self.original_bias_shape, self.shard_bias_size
                )
            self._gather_event = torch.cuda.Event()
            self._gather_event.record(self.comm_stream)

        return self._full_weights, self._full_bias

    def _gather_weights(self):
        if self._full_weights is None:
            self._start_gather()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(self._gather_event)

        self._full_weights.record_stream(current_stream)
        if self._full_bias is not None:
            self._full_bias.record_stream(current_stream)

        return self._full_weights, self._full_bias

    def _release_gather_weights(self):
        self._full_weights = None
        self._full_bias = None
        self._gather_event = None

    def _reduce_scatter(self, grad, original_rows, shard_rows, param):
        dtype = self._dtype()
        grad = grad.to(dtype).contiguous()
        total_rows = self.world_size * shard_rows

        if grad.shape[0] == total_rows:
            padded = grad
        else:
            padded = torch.zeros(
                (total_rows, *grad.shape[1:]), dtype=dtype, device=grad.device
            )
            padded[:original_rows].copy_(grad)

        local_grad = torch.empty((shard_rows, *grad.shape[1:]), dtype=dtype, device=grad.device)
        work = dist.reduce_scatter_tensor(
            local_grad, padded, op=dist.ReduceOp.SUM, async_op=True
        )

        self._pending_grad_work.append((work, local_grad, padded, param))

    def _reduce_scatter_weight_grad(self, grad):
        self._reduce_scatter(grad, self.original_shape, self.shard_size, self.weights)

    def _reduce_scatter_bias_grad(self, grad):
        self._reduce_scatter(grad, self.original_bias_shape, self.shard_bias_size, self.bias)

    def _grad_sync(self):
        for work, local_grad, _padded, param in self._pending_grad_work:
            work.wait()
            g = local_grad.to(param.dtype)  # back to FP32 master dtype
            g = g / self.world_size
            if param.grad is None:
                param.grad = g
            else:
                param.grad.add_(g)
        self._pending_grad_work.clear()

    def _prefetch_prev(self):
        owner = self._owner() if self._owner is not None else None
        if owner is not None and self._index is not None:
            owner._prefetch(self._index - 1)

    def _prefetch_next_forward(self):
        owner = self._owner() if self._owner is not None else None
        if owner is not None and self._index is not None:
            owner._prefetch(self._index + 2)