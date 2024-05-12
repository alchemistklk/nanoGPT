import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

# hyperparameter
batch_size = 32
block_size = 8
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else "cpu"
learning_rate = 1e-2
eval_interval = 300
eval_iters = 200

# ---------------

torch.manual_seed(1337)

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# convert text to chars to make dictionary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
itos = {i:ch for i, ch in enumerate(chars)}
stoi = {ch:i for i, ch in enumerate(chars)}

# encode: take string, output a list of integers
encode = lambda s: [stoi[x] for x in s]
# decode: take a list of integer, output string
decode = lambda s: ''.join([itos[x] for x in s])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(text) * 0.9)
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if 'train' == split else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.LongTensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.LongTensor(data[i+1:i+block_size+1]) for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageMode(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, x, y=None):
        # B,T,C
        logits = self.token_embedding_table(x)
        if y is None:
            loss = None
        else:
            # reshape the logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_ = self(idx)
            logits = logits[:, -1,:]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageMode(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


