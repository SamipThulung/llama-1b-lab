# load bin files that was created by the naive BPE tokenizer. 
import numpy as np
import torch

class MemmapTokenDataset:

    def __init__(
        self,
        file_path,
        seq_len,
        split,
        train_fraction=0.99,
    ):

        self.seq_len = seq_len

        # File stays on disk; no full RAM loading.
        self.tokens = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r",
        )

        self.total_tokens = len(self.tokens)

        split_idx = int(
            self.total_tokens * train_fraction
        )

        if split == "train":
            self.start_idx = 0
            self.end_idx = split_idx

        elif split == "val":
            self.start_idx = split_idx
            self.end_idx = self.total_tokens

        else:
            raise ValueError(
                "split must be train or val"
            )

        # Need seq_len input tokens + 1 target token.
        self.num_starts = (
            self.end_idx
            - self.start_idx
            - seq_len
        )

        if self.num_starts <= 0:
            raise ValueError(
                f"{split} data too small for "
                f"context_length={seq_len}"
            )

    def get_batch(self, batch_size):

        # Random contiguous windows.
        starts = np.random.randint(
            self.start_idx,
            self.start_idx + self.num_starts,
            size=batch_size,
        )

        x = np.stack([
            self.tokens[
                s : s + self.seq_len
            ]
            for s in starts
        ])

        y = np.stack([
            self.tokens[
                s + 1 : s + self.seq_len + 1
            ]
            for s in starts
        ])

        # Copy selected windows only.
        x = torch.from_numpy(
            x.astype(np.int64, copy=True)
        )

        y = torch.from_numpy(
            y.astype(np.int64, copy=True)
        )

        return (x, y)

