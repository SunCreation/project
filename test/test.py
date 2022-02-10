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
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42) 
torch.cuda.manual_seed_all(42)

homedir = input("Home DIR: ")



def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        text=examples['problem'],
        text_pair=examples['code'],
        padding='max_length',
        max_length=260
    )
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples



metric = load_metric("accuracy")

def get_answer(sent):
    sent = sent.split("<sys>")[-1]
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
        except: print("Error")
        try: label.append(input())
        except: label.append("ab")
    sys.stdout = A
    sys.stdin = B
    return {'accuracy':acsr(pred, label)}



wandb.init(project="kogpt2-pretrained-baseline", entity="math-solver")

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

dataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/Agu_bin_train.csv', split='train')
valdataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/Val_bin_train.csv', split='train')
# dictdataset = dataset.train_test_split(0.015)


tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
valtokenized_datasets = valdataset.map(prepare_train_features, batched=True, remove_columns=valdataset.column_names)

# print(tokenized_datasets)
# print(valtokenized_datasets)

print(tokenizer.decode(tokenized_datasets[0]["input_ids"]))
args = TrainingArguments(
    output_dir='kogpt-finetune',
    overwrite_output_dir = True,
    per_device_train_batch_size=14,
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
    train_dataset=tokenized_datasets,
    eval_dataset=valtokenized_datasets,
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

test = pd.read_csv(f'{homedir}/CloudData/math/data/test.csv')

import random

for _ in range(5):
    i = random.randint(0, 281)
    p = test.iloc[i]['problem']
    print(f'{p}')
    solve_problem(p)

# time.sleep(3)
# answer = input("저장 고?")
# if answer=="N": exit()

# trainer.save_model('test-kogpt-trained-hchang')