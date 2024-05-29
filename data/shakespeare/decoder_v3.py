import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.nn import functional as F


"""
Features:
1.Intersperse communication and computation
2.Residual connections()
    2.1 addition
    2.2 projection
    2.3 direct projection
3. After execution of code, train loss < val loss means a bit overfitting
    3.1 step 4800: train loss 1.9863, val loss 2.0804
4. Apply LayerNorm
    4.1 step 4800: train loss 1.9754, val loss 2.0669
5. Scaling up the model! Dropout


"""

# hyperparameter before step 5
# batch_size = 32
# block_size = 8
# max_iters = 5000
# device = "cuda" if torch.cuda.is_available() else "cpu"
# learning_rate = 1e-3
# eval_interval = 300
# eval_iters = 200
# n_embd = 32


# hyperparameter after step 5
batch_size = 64
block_size = 256
max_iters = 5000
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 3e-4
eval_interval = 200
n_embd = 384
eval_iters = 200
n_head = 6
n_layers = 6
dropout = 0.2


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
    """Single head attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, False)
        self.query = nn.Linear(n_embd, head_size, False)
        self.value = nn.Linear(n_embd, head_size, False)
        self.dropout = nn.Dropout(dropout)
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
        self.dropout(wei)
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
        # project to output
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # projection layer going back into the residual pathway
            nn.Linear(4 * n_embd, n_embd),
            # Apply dropout
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # Communication(Gather contexts)
        self.sa = MultiHeadAttention(n_head, head_size)
        # Computation(Think contexts)
        self.ffwd = FeedForward(n_embd)

        # MARK:LayerNorm before self-attention and FeedForward NetWork
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageMode(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directory reads off logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # for each token we use embed the positional information
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_embd) for _ in range(n_layers)])
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        token_embed = self.token_embedding_table(x)  # B,T,n_embd
        # input positional and token embedding
        pos_embed = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,n_embd)
        x = token_embed + pos_embed

        # Three times to communication and computation
        x = self.blocks(x)
        # Finally decode
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

for i in tqdm(range(max_iters)):
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
