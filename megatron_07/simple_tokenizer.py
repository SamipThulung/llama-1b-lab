from tokenizers import Tokenizer
class SimpleTokenizer:
    """Minimal tokenizer shim exposing what GPTDataset needs."""

    def __init__(self, path):
        self._path = path
        self._tok = Tokenizer.from_file(path)
        self.eod = self._tok.token_to_id("<|endoftext|>")
        assert self.eod is not None, "<|endoftext|> not found in tokenizer"
        self.eos = self.eod
        self.pad = self.eod
        self.vocab_size = self._tok.get_vocab_size()

    @property
    def unique_identifiers(self):
        # Used in the dataset index cache hash.
        return {"class": type(self).__name__, "path": self._path}