import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

checkpoint_dir = "/workspace/CloudData/Model/GPT_math/weight"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

checkpoint = tf.train.Checkpoint(gpt_model=model)
checkpoint.restore(latest)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = checkpoint.gpt_model


def solve_problem(problem):
    sent = problem + '<usr>'
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = model.generate(input_ids, max_length = 50, do_sample=True, top_k = 20)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    _class = sentence.split('<usr>')[1].replace('</s>', '').split('<sys>')[0].strip()
    code = sentence.split('<sys>')[1].replace('</s>', '').strip()
    #   answer = sentence.split('<sys>')[2].replace('</s>', '').strip()
    return _class, code #answer

print(solve_problem("1 더하기 일을 구하여라."))