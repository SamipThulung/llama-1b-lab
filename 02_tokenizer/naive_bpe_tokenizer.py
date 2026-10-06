import pickle
import regex
from typing import Iterable, Iterator
from config GPT2_PATTERN, GPT4_PATTERN, SPECIAL_TOKENS


class Tokenizer:
    def __init__(self ):

    self.merge_history = None
    self.id_to_bytes = None
    self.special_tokens = None
    self.vocab_size = None

    def encode(text):
        tokens = list(text.encode("utf-8"))

        for new_id, pair in merge_history.items():
            i = 0
            new_tokens = []
            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == pair
                ):
                    new_tokens.append(new_id)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens


    def decode(token_ids):
        output = b""
        for token_id in token_ids:
            output += self.id_to_bytes[token_id]
        return output.decode("utf-8")

    def from_files(self, tokenizer_file_path: str,):
        with open(tokenizer_file_path, "rb") as f:
            tokenizer_data = pickle.load(f)

        self.merge_history = tokenizer_data["merge_history"]
        self.id_to_bytes = tokenizer_data["id_to_bytes"]
        self.special_tokens = tokenizer_data["special_tokens"]
        self.vocab_size = tokenizer_data["vocab_size"]

        print("Loading complete")