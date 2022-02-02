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
# "1더하기 일을 구하시오. "

def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        text=examples['problem'],
        text_pair=examples['answer'],
        padding='max_length',
        max_length=216
    )
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples
# problem + code = input_ids
# labels <- input_ids
# input_ids -> labels
# loss scale 0.3~4
# "1더하기 일을 구하시오."


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred # , dasdas
    # print(logits.shape)
    # print(logits, labels)
    predictions = np.argmax(logits, axis=-1)
    pred = []
    label = []
    for i, j in zip(predictions, labels):
        pred.append(''.join(map(str, i)))
        label.append(''.join(map(str, j)))
    return metric.compute(predictions=pred, references=label)



# answer = []
# def compute_metrics(eval_pred):
#     output = []
#     logits, labels = eval_pred # , dasdas
#     # print(logits, labels)
#     predictions = np.argmax(logits[0], axis=-1)
#     for i in predictions:
#         output.append(code_exec(tokenizer.decode~~(i)))
#     return metric.compute(predictions=output, references=labels[0])


wandb.init(project="kogpt2-pretrained-baseline", entity="suncreation")

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

dataset = load_dataset('csv', data_files='../../CloudData/math/data/train.csv', split='train')

dictdataset = dataset.train_test_split(0.03)

# tokenizer.decode(prepare_train_features(dictdataset['train'][0])['input_ids'])

tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dictdataset["train"].column_names)
print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
args = TrainingArguments(
    output_dir='kogpt-finetune',
    overwrite_output_dir = True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs = 3,
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
    # do_eval=False
    # data_collator=data_collator,
)

# output = 
trainer.train()

# print(help(trainer.save_model))
# answer = input("저장 고?")
# if answer=="N": exit()

# trainer.save_model('test-kogpt-trained-hchang')

device = torch.device('cpu')
model = model.to(device)
# model = GPT2LMHeadModel.from_pretrained('test-kogpt-trained-hchang').to(device)

def solve_problem(problem):
    input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 216)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence = sentence.split('<pad>')[0]
    sentence = sentence.lstrip(problem)
    # sentence = sentence.split('<sys>')[1]
    # problem = sentence.split('<sys>')[0]
    sentence = sentence.lstrip()
    # print(problem)
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

