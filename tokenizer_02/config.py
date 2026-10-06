import regex
from tokenizers import Regex


# GPT2 pattern
GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+(?:\p{M}+)?|"""
    r""" ?\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|"""
    r"""\s*[\r\n]+|\s+(?!\S)|\s+"""
)

# GPT-4 pattern
GPT4_PATTERN = Regex(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

SPECIAL_TOKENS = ["<|endoftext|>","<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]

