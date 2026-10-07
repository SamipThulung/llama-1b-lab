import torch
import torch.nn.functional as F

class ShardedLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, full_weight, full_bias, shard_weight, shard_bias, layer):
        # shard_weight / shard_bias are passed in ONLY so autograd treats the
        # output as requiring grad (otherwise backward never runs). Their
        # gradients are produced via reduce-scatter in backward, so we return
        # None for them.
        y = F.linear(x, full_weight, full_bias)
        ctx.layer = layer
        ctx.has_bias = full_bias is not None
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        layer = ctx.layer
        (x,) = ctx.saved_tensors

        # Re-gather the full weights for the backward computation.
        full_weight, _ = layer._gather_weights()
        layer._prefetch_prev()

        out_features, in_features = full_weight.shape
        go = grad_output.reshape(-1, out_features).to(full_weight.dtype)
        x2 = x.reshape(-1, in_features)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = (go @ full_weight).reshape(x.shape)

        dw = go.t() @ x2
        layer._reduce_scatter_weight_grad(dw)

        if ctx.has_bias:
            db = go.sum(dim=0)
            layer._reduce_scatter_bias_grad(db)

        layer._release_gather_weights()
        return dx, None, None, None, None, None


class ShardedEmbeddingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexes, full_weight, shard_weight, layer):
        ctx.save_for_backward(indexes)
        ctx.layer = layer
        return F.embedding(indexes, full_weight)

    @staticmethod
    def backward(ctx, grad):
        (indexes,) = ctx.saved_tensors
        layer = ctx.layer
        layer._prefetch_prev()

        # No need to re-gather weights: only the shape is needed.
        flat_indexes = indexes.reshape(-1)
        flat_grad = grad.reshape(-1, layer.embedding_dim).float()

        grad_weight = torch.zeros(
            (layer.num_embeddings, layer.embedding_dim),
            dtype=torch.float32,
            device=grad.device,
        )
        grad_weight.index_add_(0, flat_indexes, flat_grad)
        grad_weight = grad_weight.to(layer._dtype())

        layer._reduce_scatter_weight_grad(grad_weight)
        layer._release_gather_weights()
        return None, None, None, None