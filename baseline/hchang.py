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
from sklearn.metrics import accuracy_score as acsr
import sys
import time


def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        text=examples['problem'],
        text_pair=examples['code'],
        padding='max_length',
        max_length=217
    )
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples



metric = load_metric("accuracy")

def get_answer(sent):
    sent = sent.split("<sys>")[1]
    sent = sent.split("<pad>")[0]
    sent = sent.strip()
    return sent

def compute_metrics(eval_pred):
    logits, labels = eval_pred 
    predictions = np.argmax(logits, axis=-1)
    pred = []
    label = []
    A = sys.stdout
    B = sys.stdin
    sys.stdout = open("stdout.txt","w")
    sys.stdin = open("stdout.txt","r")
    for i, j in zip(predictions, labels):
        i = tokenizer.decode(i)
        j = tokenizer.decode(j)
        try: exec(get_answer(i))
        except: print("error")
        try: pred.append(input())
        except: pred.append("bb")
        try: exec(get_answer(j))
        except: print("error")
        try: label.append(input())
        except: label.append("bb")
    sys.stdout = A
    sys.stdin = B
    return {'accuracy':acsr(pred, label)}



wandb.init(project="kogpt2-pretrained-baseline", entity="suncreation")

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

dataset = load_dataset('csv', data_files='../../CloudData/math/data/mytrain.csv', split='train')

dictdataset = dataset.train_test_split(0.03)


tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dictdataset["train"].column_names)
print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
args = TrainingArguments(
    output_dir='kogpt-finetune',
    overwrite_output_dir = True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs = 20,
    logging_strategy='epoch',
    save_strategy = 'epoch',
    evaluation_strategy = 'epoch',
    load_best_model_at_end = True,
    report_to="wandb"
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
    # data_collator=data_collator,
)


trainer.train()

device = torch.device('cpu')
model = model.to(device)
# model = GPT2LMHeadModel.from_pretrained('test-kogpt-trained-hchang').to(device)

def solve_problem(problem):
    input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 216)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence = get_answer(sentence)
    print('=====')
    print(f'{sentence}')
    print('실행결과:')
    try:
        exec(sentence)
    except:
        print('error')
    print("")

test = pd.read_csv('../../KMWP/data/test.csv')

import random

for _ in range(5):
    i = random.randint(0, 281)
    p = test.iloc[i]['problem']
    print(f'{p}')
    solve_problem(p)

time.sleep(3)
answer = input("저장 고?")
if answer=="N": exit()

trainer.save_model('test-kogpt-trained-hchang')