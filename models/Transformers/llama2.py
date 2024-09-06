from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import torch

import time

llama2_hf="/home/lihao/model/meta-llama/Llama-2-7b-hf"
llama2_chat_hf="/home/lihao/model/meta-llama/Llama-2-7b-chat-hf"

path = llama2_chat_hf

tokenizer = LlamaTokenizer.from_pretrained(path)

# 把 padding 讲清楚了 https://fancyerii.github.io/2024/01/04/padding/

ask1 = [
    """中国有几个省？""",
    "我今晚要去吃火锅，有什么推荐？"
]
ask2 = [
        """中国有几个省？有上海、江苏""",
        "我今晚要去吃火锅，有牛肉卷，有"
    ]
ask = ask1

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"
ids = tokenizer(ask, return_tensors='pt', padding=True)

model = LlamaForCausalLM.from_pretrained(path).cuda()

final_outputs = model.generate(
    ids["input_ids"].cuda(),
    attention_mask=ids["attention_mask"].cuda(),
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    max_length=1024,
    top_p=0.6,
    temperature=0.9,
)

output1 = tokenizer.batch_decode(final_outputs, skip_special_tokens=False)
# print("type(output1) = {}, len(output1) = {}".format(type(output1), len(output1)))
for result in output1:
    print("answers: {}".format(result))