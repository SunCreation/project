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

model = BertModel.from_pretrained("skt/kobert-base-v1")
# model = BertModel.from_pretrained("bert-base-uncased")

text = "배가 고픈데, 먹고 싶은게 없다."
text1 = "이걸로 뭘 할 수 있지? 넘 어렵다.."
tokens= tokenizer.tokenize(text1)
print(tokens)
tokens = ['[CLS]'] + tokens + ['[SEP]'] + ['[PAD]'] + ['[PAD]']

print(len(tokens))

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(token_ids)

# attention_mask = [1 if i!='[PAD]' else 0 for i in tokens]

# token_ids = th.tensor(token_ids).unsqueeze(0)
# attention_mask = th.tensor(attention_mask).unsqueeze(0)
inputs = tokenizer.batch_encode_plus([text1])



# hidden_rep, cls_head = model(token_ids,attention_mask=attention_mask)
out = model(input_ids=th.tensor(inputs['input_ids']),
            attention_mask=th.tensor(inputs['attention_mask']))
# print(hidden_rep, cls_head, sep='\n')
# print(hidden_rep.shape)
print(out.pooler_output.shape)
print(out.last_hidden_state.shape)
# print(out.__dir__())
print(model.__class__)
