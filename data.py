# common standard library utilities
import os
import sys
import glob
import time
import json
import math
import random
from random import Random

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from torch.utils.data import DataLoader, Dataset, IterableDataset

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorForLanguageModeling

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

# data utilities
import datasets
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from tqdm import tqdm

tqdm.pandas()

class ShuffleDataset(IterableDataset):
    """
    https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/
    """
    def __init__(self, dataset, buffer_size, random=random):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass

def make_dl(ds, tokenizer, batch_size=16, mlm_probability=0.15, shuffle_buffer_size=256):
    tok = ds.map(lambda x:tokenizer(x["text"], truncation=True))
    tok = tok.remove_columns(["text"])

    dl = DataLoader(
        ShuffleDataset(tok, shuffle_buffer_size, random.Random(7)),
        collate_fn = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=16 # for new tensor cores
        ),
        batch_size=batch_size
    )

    return dl

def load_dls(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    dataset = load_dataset(args.dataset, streaming=True)

    train_dl = make_dl(dataset["train"], tokenizer,
                       batch_size=args.batch_size,
                       mlm_probability=args.mlm_probability)
    val_dl = make_dl(dataset["validation"], tokenizer,
                     batch_size=args.batch_size,
                     mlm_probability=args.mlm_probability)

    return train_dl, val_dl

