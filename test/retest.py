# import tensorflow as tf
# from transformers import AutoTokenizer
# from transformers import TFGPT2LMHeadModel

# checkpoint_dir = "/workspace/CloudData/Model/GPT_math/weight"
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

# checkpoint = tf.train.Checkpoint(gpt_model=model)
# checkpoint.restore(latest)
# tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# model = checkpoint.gpt_model


# def solve_problem(problem):
#     sent = problem + '<usr>'
#     input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
#     input_ids = tf.convert_to_tensor([input_ids])
#     output = model.generate(input_ids, max_length = 50, do_sample=True, top_k = 20)
#     sentence = tokenizer.decode(output[0].numpy().tolist())
#     _class = sentence.split('<usr>')[1].replace('</s>', '').split('<sys>')[0].strip()
#     code = sentence.split('<sys>')[1].replace('</s>', '').strip()
#     #   answer = sentence.split('<sys>')[2].replace('</s>', '').strip()
#     return _class, code #answer

# print(solve_problem("1 더하기 일을 구하여라."))

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
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        text=examples['problem'],
        text_pair=examples['code'],
        padding='max_length',
        max_length=216
    )
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples

dataset = load_dataset('csv', data_files='../../CloudData/math/data/train.csv', split='train')

dictdataset = dataset.train_test_split(0.03)

test = pd.read_csv('../../KMWP/data/test.csv')

import random

def solve_problem(problem):
    input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 216)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence = sentence.split('<pad>')[0]
    sentence = sentence.lstrip(problem)
    sentence = sentence.lstrip(' ')
    print('=====')
    print(f'{sentence}')
    print('실행결과:')
    try:
        exec(sentence)
    except:
        print('error')
    print("")



for _ in range(5):
    i = random.randint(0, 281)
    p = test.iloc[i]['problem']
    print(f'{p}')
    solve_problem(p)
