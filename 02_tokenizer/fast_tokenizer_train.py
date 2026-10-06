from itertools import islice
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from config import GPT4_PATTERN, SPECIAL_TOKENS

VOCAB_SIZE = 32000
N_DOCS = 1_000_000
output_file = "fineweb_bpe_32k.json"

from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train", streaming=True)

def text_iterator():
    for ex in islice(ds, N_DOCS):
        yield ex["text"]

tok = Tokenizer(models.BPE())  
tok.normalizer = normalizers.NFC()

tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(GPT4_PATTERN, behavior="isolated", invert=False),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,                       # includes special tokens
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes guaranteed
    min_frequency=2,
    show_progress=True,
)

tok.train_from_iterator(text_iterator(), trainer=trainer, length=N_DOCS)
tok.save(output_file)
print(tok.get_vocab_size())  # 32000