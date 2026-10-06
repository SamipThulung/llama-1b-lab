from tokenizers import Tokenizer

tok = Tokenizer.from_file("fineweb_bpe_32k.json")

enc = tok.encode("Hello world, this is FineWeb 2024!")
print(enc.tokens)   
print(enc.ids)     
print(tok.decode(enc.ids))
print(tok.get_vocab_size())  