import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
import pandas as pd
import torch
# checkpoint_dir = "/workspace/CloudData/Model/GPT_math/weight"
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
device = torch.device('cpu')
# model = model.to(device)
model = GPT2LMHeadModel.from_pretrained("kogpt-finetune/checkpoint-6224").to(device) # 'test-kogpt-trained-hchang'

# checkpoint = tf.train.Checkpoint(gpt_model=model)
# checkpoint.restore(latest)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# model = checkpoint.gpt_model
data = pd.read_csv("../../CloudData/math/data/test.csv")

def get_answer(sent):
    sent_ = sent.split("<sys>")[-1]
    class_ = sent.split("<sys>")[1]
    sent = sent_.split("<pad>")[0]
    sent = sent.strip()
    return sent, class_

def solve_problem(problem, i):
    input_ids = tokenizer(problem+"<sys>",return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence, class_ = get_answer(sentence)
    # print(problem.rstrip("<sys>"))
    # print('{')
    print(str(i+1) + ':')
    # print('=====')
    problem = problem.replace('"', "'")
    print(f'  class: {class_}')
    print(f'  problem: "{problem}"')
    newsentence = sentence.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n\n').replace('"', "'")
    print(f'  code: "{newsentence}"')
    # print('실행결과:')
    try:
        print("  answer:",end=' ')
        exec(sentence)
    except:
        print('error')
    print("")
# for i in data["problem"][:5]:
#     solve_problem(i)

import sys
from tqdm import tqdm
stdout_ = sys.stdout
sys.stdout = open("test.yaml", 'w')
t = tqdm(range(len(data)))
for i in t:
    j = data['problem'][i]
    solve_problem(j, i)

sys.stdout = stdout_

import yaml
import json
from collections import defaultdict
with open("test.yaml", 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
print(data)
result =defaultdict(dict)
for i in range(len(data)):
    result[str(i+1)] = data[i+1]
jstring = json.dumps(result, indent=4)
with open("test.json", "w") as f:
    f.write(jstring)


# solve_problem("정배님은 회식비로 20만원을 지불했다. 식사비가 15만원 들었을때, 남는 돈을 구하여라.")
# import os
# from os import path
# import transformers
# import torch
# import wandb
# from transformers import (
#     AutoTokenizer, 
#     GPT2LMHeadModel,
#     TrainingArguments,
#     Trainer,
# )
# from datasets import load_dataset
# import pandas as pd
# from datasets import load_metric
# import numpy as np


# device = torch.device('cpu')

# model = GPT2LMHeadModel.from_pretrained('test-kogpt-trained-hchang').to(device)
# tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

# def prepare_train_features(examples):
#     tokenized_examples = tokenizer(
#         text=examples['problem'],
#         text_pair=examples['code'],
#         padding='max_length',
#         max_length=216
#     )
#     tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
#     return tokenized_examples





# dataset = load_dataset('csv', data_files='../../CloudData/math/data/train.csv', split='train')

# dictdataset = dataset.train_test_split(0.03)

# tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dictdataset["train"].column_names)
# print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids']))

# test = pd.read_csv('../../KMWP/data/test.csv')

# import random

# def solve_problem(problem):
#     input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
#     output = model.generate(input_ids, max_length = 216)
#     sentence = tokenizer.decode(output[0].numpy().tolist())
#     sentence = sentence.split('<pad>')[0]
#     sentence = sentence.lstrip(problem)
#     sentence = sentence.lstrip(' ')
#     print(problem)
#     print('=====')
#     print(f'{sentence}')
#     print('실행결과:')
#     try:
#         exec(sentence)
#     except:
#         print('error')
#     print("")



# # for _ in range(5):
# #     i = random.randint(0, 281)
# #     p = test.iloc[i]['problem']
# #     print(f'{p}')
# #     solve_problem(p)
# solve_problem('지희는 밑변이 5센티미터 이고 높이가 6센티미터 인 삼각형을 그렸고, 동권이는 밑변이 12센티미터 이고 높이가 5센티미터 인 삼각형을 그렸습니다. 동권이가 그린 삼각형의 넓이는 지희가 그린 삼각형의 넓이의 몇 배입니까?')