
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

def build_llama_model(model_detail):
    num_layers = model_detail["num_layers"]
    hidden = model_detail["d_model"]
    heads = model_detail["num_heads"]

    assert VOCAB_SIZE % TP_SIZE == 0, "vocab size must be divisible by TP size"

    # Llama-style FFN
    ffn = int(8 * hidden / 3) // 64 * 64

    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        ffn_hidden_size=ffn,

        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,

        attention_dropout=0.0,
        hidden_dropout=0.0,

        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=PP_SIZE,
        sequence_parallel=False,

        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )

    model = GPTModel(
        config=config,
        # normalization arg selects RMSNorm in the local spec
        # (needs a Megatron-Core version that supports it, and torch >= 2.4).
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=CONTEXT_LENGTH,
        pre_process=True,
        post_process=True,
        # Logits stay TP-sharded; loss uses vocab-parallel cross entropy.
        parallel_output=True,
        position_embedding_type="rope",
    )

    return model
