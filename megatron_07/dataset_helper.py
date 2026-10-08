
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

def build_dataloader():
    tokenizer = SimpleTokenizer(TOKENIZER_PATH)
    assert tokenizer.vocab_size == VOCAB_SIZE, (
        f"tokenizer vocab {tokenizer.vocab_size} != VOCAB_SIZE {VOCAB_SIZE}"
    )

    config = GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=CONTEXT_LENGTH,
        blend=([DATASET_FILEPATH], None),   # single dataset, no weights
        split="98,1,1",
        path_to_cache=CACHE_PATH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
    )

    # Sizes are NUMBER OF SAMPLES (sequences), not tokens.
    train_samples = NUM_ITERATIONS * NUM_MICROBATCHES * MICRO_BATCH_SIZE * DP_SIZE
    sizes = [train_samples, 1_000, None]

    # Every rank builds/loads the datasets (rank 0 builds the index first).
    # TP ranks must see identical data, so no TP-dependent logic here.
    train_ds, _valid_ds, _test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, sizes, lambda: True, config
    ).build()

    # GPTDataset already handles shuffling internally, so shuffle=False.
    # The sampler splits data across DP ranks; TP ranks in the same DP
    # group share the same dp_rank and therefore get identical batches.
    sampler = DistributedSampler(
        train_ds,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=False,
        drop_last=True,
    )

    return DataLoader(
        train_ds,
        batch_size=MICRO_BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

