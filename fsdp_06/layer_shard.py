from fsdp_06.shard_base import SharedBase
from fsdp_06.custom_autograd_functions import ShardedLinearFunc, ShardedEmbeddingFunc

class ShardedLinear(SharedBase):
    def __init__(self, original: nn.Linear, rank, world_size, compute_dtype):
        super().__init__(rank, world_size, compute_dtype)
        self.in_features = original.in_features
        self.out_features = original.out_features

        self.weights, self.original_shape, self.shard_size = self._create_shard(original.weight)
        if original.bias is not None:
            self.bias, self.original_bias_shape, self.shard_bias_size = self._create_shard(original.bias)

    def forward(self, x):
        if self.compute_dtype is not None and x.dtype != self.compute_dtype:
            x = x.to(self.compute_dtype)  # outside the Function so autograd handles it

        full_weight, full_bias = self._gather_weights()
        y = ShardedLinearFunc.apply(x, full_weight, full_bias, self.weights, self.bias, self)
        self._release_gather_weights()
        self._prefetch_next_forward()
        return y


class ShardedEmbedding(SharedBase):
    def __init__(self, original: nn.Embedding, rank, world_size, compute_dtype):
        super().__init__(rank, world_size, compute_dtype)
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.weights, self.original_shape, self.shard_size = self._create_shard(original.weight)

    def forward(self, indexes):
        full_weight, _ = self._gather_weights()
        y = ShardedEmbeddingFunc.apply(indexes, full_weight, self.weights, self)
        self._release_gather_weights()
        self._prefetch_next_forward()
        return y