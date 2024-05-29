import os
from tqdm import tqdm
import numpy as np
import tiktoken
import random
from datasets import load_dataset

# number of workers in .map() call
# good number to use is ~order number of cpu cores
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on network speed
num_proc_load_dataset = num_proc

source_data_path = (
    "/Users/alchemist/Projects/huggingface/openwebtext/urlsf_subset00-1_data"
)

# get encoding
enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":

    # 1. Load data from disk to memory
    # Traverse all subdirectories to get all txt files
    data_path_list = []
    for root, _, files in os.walk(source_data_path):
        for file in files:
            data_path_list.append(os.path.join(root, file))

    # Randomly choose a file as text file
    val_idx = random.randint(0, len(data_path_list) - 1)
    val_file_path = data_path_list[val_idx]
    data_path_list.pop(val_idx)

    # Load all txt files to dataset
    dataset = load_dataset(
        "text", data_files={"train": data_path_list, "val": val_file_path}
    )

    # 2. Tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        # add the end of text token
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len':len(ids)}
        return out
    
    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc='tokenizing the splits',
        num_proc=num_proc
    )
    
    # concatenate all ids in each dataset into a large file for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename=filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 32
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        