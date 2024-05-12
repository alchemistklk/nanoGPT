import os
import numpy as np
import requests
import tiktoken

# declare the file path
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

# download the file
if not os.path.exists(input_file_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)

# read the content of file
with open(input_file_path, "r", encoding="utf-8") as f:
    text = f.read()
n = len(text)
train_data = text[: int(0.9 * n)]
val_data = text[int(0.9 * n) :]

# 1.tokenizer(easy mapping)
# here is a easy implementation of tokenizer
# create a mapping from characters to integers
# here are all unique characters that occurt in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)
# string to integer
# stoi = {ch:i for i, ch in enumerate(chars)}
# # integer to string
# itos = {i:ch for i, ch in enumerate(chars)}
# encode = lambda s: [stoi[x] for x in s]
# decode = lambda s: ''.join([itos[x] for x in s])

# print(encode("hii there"))
# print(decode(encode("hii there")))

# 2.tokenizer(tiktoken)
# encoder with tiktoken gpt2 here
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids)} tokens")
print(f"val has {len(val_ids)} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
