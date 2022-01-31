# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
import os
from os import path
import transformers
import torch
import wandb
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import pandas as pd
from datasets import load_metric
import numpy as np


device = torch.device('cpu')

model = GPT2LMHeadModel.from_pretrained('test-kogpt-trained-hchang').to(device)


def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        text=examples['problem'],
        text_pair=examples['code'],
        padding='max_length',
        max_length=216
    )
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples

dataset = load_dataset('csv', data_files='../CloudData/math/data/train.csv', split='train')

dictdataset = dataset.train_test_split(0.03)

test = pd.read_csv('../../KMWP/data/test.csv')

import random

for _ in range(5):
    i = random.randint(0, 281)
    p = test.iloc[i]['problem']
    print(f'{p}')
    solve_problem(p)
