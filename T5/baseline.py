
import pandas as pd
dataset = pd.read_csv('../../CloudData/math/data/mytrain.csv')



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import YourDataSetClass
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-small")

model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-small")


# the following 2 hyperparameters are task-specific
# max_source_length = 512
# max_target_length = 128

# # Suppose we have the following 2 training examples:
sentences = dataset[dataset["class"]==1]["problem"]
output_sequence_1 = dataset[dataset["class"]==1]["code"]


# encode the inputs
task_prefix = "산술연산 파이썬 코드로: "

inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)


from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'











output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)
print(sentences)
print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

