# # import transformers
# # import torch
# # import numpy

# # from transformers import (
# #     AutoTokenizer,
# #     AutoConfig,
# #     AutoModelForSeq2SeqLM,
# #     DataCollatorForSeq2Seq,
# #     EarlyStoppingCallback,
# #     Seq2SeqTrainingArguments,
# #     Seq2SeqTrainer
# # )
# # from datasets import load_dataset
# # import pandas as pd

# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-small-ko", use_fast = False)
# # model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-small-ko")

# import pandas as pd

# data = pd.read_csv("../../CloudData/math/data/train.csv")
# # 예진님의 나이를 구하시오. <sys> 9 <sys> code
# # def datamo(x):
# #     for i in data["class"]:
# #         output = x + '<sys>' + str(i) + '<sys>'
# #         yield output

# # def test(x):
# #     print(x)
# #     return x 
# def datamf(x):
#     for i in data["code"]:
#         y = x + '<sys>' + str(i) + '<sys>'
#         yield y


# data['problem'] + data['code']
# data['problem'] = data['problem'].apply(lambda x: x + '<sys>')
# # data = data.apply(test)

# for i in range(len(data["problem"])):
#     data["problem"][i] = data['problem'][i] + '<sys>' + str(data['class'][i]) + '<sys>'

# # print(data["problem"][0])
# # print(data['problem'])
# # for i in data.values:
# #     print(i[0])
# #     break

# data.to_csv("../../CloudData/math/data/newtrain.csv")


import sys
sys.stdout = open("test2.yaml", 'w')
print("hihi")