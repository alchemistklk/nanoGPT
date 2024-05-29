import math

import torch
import inspect
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """An LayerNorm module with an optional bias parameter."""

    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x):
        x = F.layer_norm(
            x, self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5
        )


class CasualSelfAttention(nn.Module):
    """The implementation of self-attention mechanism in GPT model"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        # output the projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        # flash attention to make GPU brrrrr
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # casual mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_length, n_embed
        # calculate query, key, values for all heads in batch and move head forward to be the second dimension
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_h, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_h, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_h, T, hs)

        # casual self-attention; Self-attention (B, n_h, T, hs) x (B, n_h, hs, T) -> (B, n_h, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of casual self-attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_h, T, T) @ (B, n_h, T, hs) -> (B, n_h, T, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(self.resid_dropout(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.attn = CasualSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True  # True bias in Linears and LayerNorms, like gpt2


class GPT(nn.Module):
    def __init__(self, config):
        super.__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # word token embedding.
                # Convert input token IDs(which are integer representation of words or sub-words) into dense vector representation
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                # word position embedding
                # Help the model understand the structure and meaning based on the position of tokens in the sequence
                wpe=nn.Embedding(config.block_size, config.n_embed),
                drop=nn.Dropout(config.dropout),
                # contains several blocks
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embed, bias=config.bias),
            )
        )

        # projection the final embedding to vocab
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projection, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj_weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print(f"number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding=True):
        """Return the parameters in the model
        For non-embedding count(default), this position embeddings get subtracted
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # token embedding of (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos)  # token embedding of (t, n_embed)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits, target)
        else:
            # inference-time: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = (
            override_args or {}
        )  # if override_args is None, set it to an empty dictionary

        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_embed, n_head
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        print("forcing vocab size to 50257, block_size to 1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints

        if "dropout" in override_args:
            print("overriding dropout to %.3f" % override_args["dropout"])
            config_args["dropout"] = override_args["dropout"]

        # create a from-scratch model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init a huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        # ignore the buffer
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.bias")]

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_hf_keys) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_hf_keys)} != {len(sd_keys)}"

        for k in sd_hf_keys:
            if (k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate the model flops utilization (MFU) in units if A100 bfloat16 flops."""
        # First we need to estimate the number of flops we do per iteration
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embed // cfg.n_head, cfg.block_size
        # calculate the number of flops per token
        flops_per_token = 6 * N + 12 * L * H * Q * T
        # calculate the number of flops per forward-backward pass
        flops_per_fwdbwd = flops_per_token * T
        # calculate the number of flops per batch_size
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput in A100 bfloat16 flops
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312 * 10**12  # 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Taking a conditioning sequence of indices idx(LongTensor of shape (b, t)) and generating max_new_tokens times
        feeding the predictions back into the model each time.
        Args:
            idx (_type_): LongTensor of shape (b, t)
            max_new_tokens (_type_): sequence length to generate
            temperature (float, optional): _description_. Defaults to 1.0.
            tok_k (_type_, optional): _description_. Defaults to None.
        """

        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # set all logits to -inf that are not in the top_k
                logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax
                probs = F.softmax(logits, dim=-1)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

            return idx


def configure_optimizer(self, lr, weight_decay, betas, device_type):
    """Introduction to the function
    1. Extract all parameters that require grad
    2. Create two groups of parameters: one for weight decay and one for no weight decay
    3. Create the optimizer with the two groups
    4. Print the number of parameters in each group
    5. Use the fused version of the optimizer if available
    6. Initialize the optimizer with the parameters
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those don't require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups, any parameters in 2D will be weight decayed, otherwise no.
    # all weight tensors in matmul + embeddings decay, all bias and layernorms tensors don't
    decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
    nondecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nondecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nondecay_params = sum(p.numel() for p in nondecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params:,} parameters"
    )

    # Create AdamaW optimizer and use fused version of it if available
    fused_available = "fused" in inspect.getsource(torch.optim.AdamW).parameters
    device_type = torch.cuda.get_device_name(0) if device_type == "cuda" else "cpu"
    used_fused = fused_available and device_type == "cuda"
    extra_args = dict(fuse=True if used_fused else False)
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
    print(f"using {'fused' if used_fused else 'python'} adamw optimizer")
    return optimizer
