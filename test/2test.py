from datasets import load_dataset

raw_sets = load_dataset("imdb")
print(raw_sets["train"]['text'][0],raw_sets["train"]['label'][0])
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# inputs = tokenizer(sentences, padding="max_length", truncation=True)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) 
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) 
# full_train_dataset = tokenized_datasets["train"]
# full_eval_dataset = tokenized_datasets["test"]