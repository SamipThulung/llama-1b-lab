
from megatron.core.transformer.module import Float16Module

def wrap_model_for_distributed_training(model):
    # Float16Module (as in Megatron's own training code) handles bf16 casting.
    model = Float16Module(model.config, model)

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,       # accumulate / reduce grads in fp32
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )

    model = DistributedDataParallel(
        config=model.config,
        ddp_config=ddp_config,
        module=model,
    )
    return model


def build_optimizer(model):
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=LR,
        min_lr=MIN_LR,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=0.9,
        adam_beta2=0.95,
        decoupled_weight_decay=True,    # AdamW
        bf16=True,
        clip_grad=GRAD_CLIP,
        use_distributed_optimizer=False,
    )

    return get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[model],
    )