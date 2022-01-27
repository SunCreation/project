print("helloworld")
import torch as th
from transformers import (BertModel, 
BertTokenizer, BertForSequenceClassification, BertTokenizerFast,
Trainer, TrainingArguments)
from kobert_tokenizer import KoBERTTokenizer
import numpy as np
from nlp import load_dataset

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

what = tokenizer.encode("한국어 모델을 공유합니다.")
print(what)

# model = BertModel.from_pretrained("skt/kobert-base-v1", output_hidden_states=True)
# model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)

# text = "배가 고픈데, 먹고 싶은게 없다."
text1 = "이걸로 뭘 할 수 있지? 넘 어렵다.."
# tokens= tokenizer.tokenize(text1)
# print(tokens)
# tokens = ['[CLS]'] + tokens + ['[SEP]'] + ['[PAD]'] + ['[PAD]']

# print(len(tokens))

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(token_ids)

# attention_mask = [1 if i!='[PAD]' else 0 for i in tokens]

# token_ids = th.tensor(token_ids).unsqueeze(0)
# attention_mask = th.tensor(attention_mask).unsqueeze(0)
inputs = tokenizer.batch_encode_plus([text1])



# hidden_rep, cls_head = model(token_ids,attention_mask=attention_mask)
# out = model(input_ids=th.tensor(inputs['input_ids']),
#             attention_mask=th.tensor(inputs['attention_mask']))
# print(hidden_rep, cls_head, sep='\n')
# print(hidden_rep.shape)
# print(out.pooler_output.shape)
# print(out.last_hidden_state.shape)
# print(out.last_hidden_state[0][0])
# print(out.hidden_states[0])


# print(out.__dir__())
# print(model.__class__)
# -------------------데이터 받기-------------------
dataset = load_dataset("csv", data_files="imdbs.csv", split="train")
print(type(dataset))

dataset = dataset.train_test_split(test_size=0.2)
print(dataset)

train_set = dataset["train"]
test_set = dataset["test"]


print(train_set, train_set['text'][0],train_set['label'][:5], sep='\n')

# ----------------- 데이터 전처리--------------------
def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True)

train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

train_set.set_format("torch", columns=['input_ids','attention_mask','label'])
test_set.set_format("torch", columns=['input_ids','attention_mask','label'])

# --------------------학습 ---------------------------

batch_size = 8
epochs= 2 

warmup_steps = 500
weight_decay = 0.01

training_args = TrainingArguments(
    output_dir = 'results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    # evaluate_during_training=True,
    logging_dir='logs'
)

model_name_or_path = 'bert-base-uncased'
with training_args.strategy.scope():    # training_args가 영향을 미치는 model의 범위를 지정
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
# model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set
)

trainer.train()

# trainer.evaluate()

