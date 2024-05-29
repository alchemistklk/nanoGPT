"""
A much more shorter version of training.py for benchmarking purposes.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
from model import GPTConfig, GPT


# ----------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
profile = False  # use pytorch profiler, or just simple benchmarking?
# ----------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# MARK: Several ways to enable Tensor Cores on Ampere GPUs
# Enable TF32 for matrix multiplications on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
# Allow cuDNN to use Tensor Cores for faster training
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
# MARK: By Using this approach, you can flexibly switch between different device and data type
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# data loading it
if real_data:
    dataset = "openwebtext"
    data_dir = os.path.join("data", dataset)
    # MARK: Load the data from disk to memory, especially in `.bin` extension file
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )

    def get_batch(split):
        data = train_data
        idx = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack(
            torch.from_numpy(data[i : i + block_size]).astype(np.int64) for i in idx
        )
        y = torch.stack(
            torch.from_numpy(data[i + 1 : i + block_size + 1]).astype(np.int64)
            for i in idx
        )
        # MARK: Pin memory in RAM for faster data transfer
        x, y = x.pin_memory(device, non_blocking=True), y.pin_memory(
            device, non_blocking=True
        )
        return x, y

else:
    # MARK: Randomly generate some data for benchmarking
    x = torch.randint(0, 50256, (batch_size, block_size), device=device)
    y = torch.randint(0, 50256, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

gptconf = GPTConfig(
    block_size=block_size,  # how far back does the model look? i.e. context size
    n_layer=12,
    n_head=12,
    n_embd=768,  # size of the model
    dropout=0,  # for determinism
    bias=bias,
)

model = GPT(gptconf).to(device, dtype=ptdtype)
optimizer = model.configure_optimizer(
    weight_decay=0.01, lr=1e-4, betas=(0.9, 0.95), device_type=device_type
)

# MARK: Print the model
if compile:
    model = torch.compile(model)

if profile:
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/bench_log"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
        with_modules=False, # only for torchscript model atm
    ) as prof:
        for i in range(num_steps):
            x, y = get_batch("train")
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(x, y)
                # MARK: split the data again 
                x, y = get_batch("train")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"step {k}/{num_steps}, loss: {lossf:.4f}")
                prof.step()

else:
    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]):
        t0 = time.time()
        X, Y = get_batch("train")
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch("train")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"stage {stage}, step {k}/{num_steps}, loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        # TODO: Implement the method of estimate mfu
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")