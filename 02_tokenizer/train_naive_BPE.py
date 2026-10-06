# Training a naive BPE Tokenizer on TinyStories dataset
# targeted vocab size = 10000
import os
import time
import regex
from collections import Counter
import pickle
from config import GPT2_PATTERN
from naive_bpe_helper import find_chunk_boundaries, pretokenize_and_count


filename = "TinyStories-train.txt"
num_chunks = 8

target_vocab_size = 10000
new_id = 256

output_file = "naive_tokenizer.pkl"


start_time = time.time()

print("Finding chunk boundaries")

with open(filename, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

print(f"Boundaries found: {len(boundaries)}")
print(f"Actual chunks:{len(boundaries) - 1}")
print()



print("Pretokenization")

word_freq = pretokenize_and_count(filename,boundaries)
end_time = time.time()

print(f"\nPretokenization time: {end_time - start_time:.2f} seconds")



print("Creating initial vocabulary")

vocab = Counter()
for word, freq in word_freq.items():

    vocab[tuple(word.encode("utf-8"))] += freq

print(f"Initial vocabulary sequences: {len(vocab):,}")

del word_freq


print("Initial vocabulary created.")

print("BPE training")
id_to_bytes = {
    i: bytes([i])
    for i in range(256)
}

merge_history = {}
training_start = time.time()

while new_id < target_vocab_size:

    pairs = get_pairs(vocab)

    if not pairs:
        break

    best_pair = max(
        pairs,
        key=pairs.get
    )

    best_pair_frequency = pairs[
        best_pair
    ]

    merge_history[
        new_id
    ] = best_pair

    id_to_bytes[
        new_id
    ] = (
        id_to_bytes[
            best_pair[0]
        ]
        +
        id_to_bytes[
            best_pair[1]
        ]
    )

    vocab = merge_pairs(
        vocab,
        best_pair,
        new_id
    )

    new_id += 1

    print("BPE TRAINING")
    print(f"Current vocabulary size:{new_id:,}")
    print(f"Target vocabulary size: {target_vocab_size:,}")
    print(f"Progress:{new_id / target_vocab_size * 100:.2f}%")
    print(f"Last merged pair: {best_pair}")
    print(f"Pair frequency: {best_pair_frequency:,}")
    print(f"Last token byte length:{len(id_to_bytes[new_id - 1])}")
    print(f"Merges completed: {len(merge_history):,}")

training_end = time.time()


print()
print("BPE TRAINING COMPLETE")
print()
print(f"Final vocabulary size: {new_id:,}")
print(f"Number of merges: {len(merge_history):,}")
print(f"Training time: {training_end - training_start:.2f} seconds")
print(f"Total time: {training_end - start_time:.2f} seconds")

print()
print("Longest token_length")
longest_id = max(id_to_bytes, key=lambda i: len(id_to_bytes[i]))
print("Token ID:", longest_id)
print("Token bytes:", id_to_bytes[longest_id])
print("Token length:", len(id_to_bytes[longest_id]))
print("Token text:", id_to_bytes[longest_id].decode("utf-8", errors="replace"))
print("Top 20 longest token")
for token_id, token_bytes in sorted(id_to_bytes.items(), key=lambda item: len(item[1]), reverse=True)[:10]:
    print(token_id, len(token_bytes), repr(token_bytes))



print()
print("Saving Tokenizer")

tokenizer_data = {
    "merge_history": merge_history,
    "id_to_bytes": id_to_bytes,
    "special_tokens": ["<|endoftext|>"],
    "vocab_size": new_id,
}

with open(output_file, "wb") as f:
    pickle.dump(tokenizer_data, f)
print(f"\nTokenizer saved to {output_file}")

