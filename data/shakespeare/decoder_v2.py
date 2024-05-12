import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

# hyperparameter
batch_size = 32
block_size = 8
max_iters = 5000
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3
eval_interval = 300
eval_iters = 200
n_embd = 32

# ---------------

torch.manual_seed(1337)

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

with open(input_file_path, "r", encoding="utf-8") as f:
    text = f.read()

# convert text to chars to make dictionary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
itos = {i: ch for i, ch in enumerate(chars)}
stoi = {ch: i for i, ch in enumerate(chars)}

# encode: take string, output a list of integers
encode = lambda s: [stoi[x] for x in s]
# decode: take a list of integer, output string
decode = lambda s: "".join([itos[x] for x in s])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(text) * 0.9)
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if "train" == split else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.LongTensor(data[i : i + block_size]) for i in ix])
    y = torch.stack([torch.LongTensor(data[i + 1 : i + block_size + 1]) for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Single head attention

    Args:
        nn (_type_): _description_
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, False)
        self.query = nn.Linear(n_embd, head_size, False)
        self.value = nn.Linear(n_embd, head_size, False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B, T, C
        q = self.query(x)  # B, T, C
        v = self.value(x)  # B, T, C

        wei = q @ k.transpose(-2, -1) * C**-0.5  # B, T, T
        # masked
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # B, T, T
        out = wei @ v
        return out  # B, T, T @ B, T, C -> B, T, C


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel

    Args:
        num_heads: number of heads
        head_size: size of each head
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity
    
    """
    def __init__(self, n_embd):
        super().__init__()
        # MARK: FeedForward is per token level,
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


class BigramLanguageMode(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directory reads off logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # for each token we use embed the positional information
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = MultiHeadAttention(4, n_embd // 4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        token_embed = self.token_embedding_table(x)  # B,T,n_embd
        # input positional and token embedding
        pos_embed = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,n_embd)
        x = token_embed + pos_embed

        # self-attention
        x = self.sa_head(x)
        # MARK:Think what they found from other tokens, use this
        x = self.ffwd(x)
        logits = self.lm_head(x)  # B,T,vocab_size

        if y is None:
            loss = None
        else:
            # reshape the logits
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # crop idx to the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageMode()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
