import os
import pandas as pd
from transformers import pipeline
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
# from tensorflow.compat.v1 import ConfigProto
# from numba import cuda

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
from core import LearningRateScheduler


# config = ConfigProto()
# config.gpu_options.allow_growth = True



inputs = tokenizer("짱짱", return_tensors="tf")
sample = tokenizer.tokenize("예진님 짱짱")
outputs = model(inputs)
print(model.config)


# print(tokenizer.cls_token)
answer = input("Enter를 쳐주세요.(멈추려면 N)")
if answer == "N":exit()

data_path = "/workspace/CloudData/math/data/train.csv"

train_data = pd.read_csv(data_path)
print(train_data.head(3))

BATCH_SIZE = 10

def get_math_data():
    for _class, problem, code, ans in zip(train_data["class"][:-100], train_data.problem.to_list()[:-100], train_data.code.to_list()[:-100], train_data["answer"][:-100]):
        bos_token = [tokenizer.bos_token_id]
        eos_token = [tokenizer.eos_token_id]
        sent = tokenizer.encode(problem+'<sys>' + code)
        yield bos_token + sent + eos_token

dataset = tf.data.Dataset.from_generator(get_math_data, output_types=tf.int32)

dataset = dataset.padded_batch(batch_size=BATCH_SIZE, padded_shapes=(None,), padding_values=tokenizer.pad_token_id)

for batch in dataset:
#   print(batch)
    break

print(tokenizer.decode(batch[0]))


learning_rate=LearningRateScheduler(768, 400)
adam = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-08)

steps = len(train_data) // BATCH_SIZE + 1
# print(steps)

print(batch,type(batch))

_len = len(train_data)-100

# answer = input("Enter를 쳐주세요.(멈추려면 N)")
# if answer == "N":exit()

# --------------------------
# def get_math_data():
#     for _class, problem, code in zip(train_data["class"], train_data.problem.to_list(), train_data.code.to_list()):
#         bos_token = [tokenizer.bos_token_id]
#         eos_token = [tokenizer.eos_token_id]
#         sent = tokenizer.encode(str(_class)+' <usr>'+problem+'<sys>'+code) # str(_class)+
#         yield bos_token + sent + eos_token

# model.train()



def solve_problem(problem):
    sent = problem + '<sys>'
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = model.generate(input_ids, max_length = 50, do_sample=True, top_k = 20)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    # _class = sentence.split('<usr>')[1].replace('</s>', '').split('<sys>')[0].strip()
    code = sentence.split('<sys>')[1].replace('</s>', '').strip()
    # answer = sentence.split('<sys>')[2].replace('</s>', '').strip()
    return code #answer

test_path = "/workspace/CloudData/math/data/test.csv"

test_data = pd.read_csv(test_path)

from tqdm import tqdm
from IPython import display

fit = True
display.clear_output(wait=True)
while fit:
    EPOCHS = int(input("학습시작, 몇에폭? : "))

    for epoch in range(EPOCHS):
        epoch_loss = 0

        idx_list = list(range(0, _len, BATCH_SIZE))[:-1]
        # random.shuffle(idx_list)
        t = tqdm(idx_list)
        nowdata = dataset.__iter__()

        for batch,_ in enumerate(t):
            with tf.GradientTape() as tape:
                now = next(nowdata)
                result = model(now, labels = now)
                loss = result[0]
                
                # print(help(result))
                batch_loss = tf.reduce_mean(loss)
            
            grads = tape.gradient(batch_loss, model.trainable_variables)
            adam.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += batch_loss
            t.set_description_str('Epoch %2d' % (epoch + 1))    
            t.set_postfix_str('Loss %.4f' % (epoch_loss / (batch + 1)))
            # cuda.close()

    for i in test_data["problem"][:5]:
        print(i)
        print("=====================")
        print(solve_problem(i))
        print("")
    answer = input("학습을 마칠까요?(아니면 N): ")
    if answer!="N": fit=False



        # for i in 


# - ---------------------------------

# for (batch, idx) in enumerate(t):
#     batch_loss, enc_attns, dec_attns, dec_enc_attns = \
#     train_step(x_train[idx:idx+BATCH_SIZE],
#                     y_train[idx:idx+BATCH_SIZE],
#                     optimizer)

#     total_loss += batch_loss
    
#     t.set_description_str('Epoch %2d' % (offset_epoch+epoch + 1))
#     t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
#     history['loss'].append(total_loss.numpy() / (batch + 1))








# print(test_data.head())



answer = input("저장하시겠습니까?(아니면 N):")
# tf.Checkpoint
checkpoint_dir = "/workspace/CloudData/Model/GPT_math/weight"
if answer!="N":
    mname = input("모델 이름: ")
    # model_path = os.path.join(model_dir, f"{mname}.h5")
    # model.save_weights(model_path)
    checkpoint_prefix = os.path.join(checkpoint_dir, mname)
    checkpoint = tf.train.Checkpoint(gpt_model=model)

    checkpoint.save(file_prefix=checkpoint_prefix)
    print("저장완료")
