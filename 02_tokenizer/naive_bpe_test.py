import pickle
from naive_bpe_tokenizer import Tokenizer

tokenizer = Tokenizer()
tokerizer.from_files("naive_tokenizer.pkl")

sentence = "Hello, how are you today?, how are the cat"

token_ids = tokenizer.encode(sentence)
print("Original:")
print(sentence)
print("Token IDs:")
print(token_ids)

retokenized = tokenizer.decode(token_ids)

print("Decoded:")
print(retokenized)
print("Sanity check:")

if sentence == retokenized:
    print("PASS: original == decoded")
else:
    print("FAIL: original != decoded")