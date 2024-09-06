from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import torch

import time

#tokenizer = AutoTokenizer.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b")
#model = AutoModelForCausalLM.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b").cuda()

#tokenizer = LlamaTokenizer.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b_convert/")
#model = LlamaForCausalLM.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b_convert/").cuda()

tokenizer = LlamaTokenizer.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b-hf").cuda()

model.generation_config.pad_token_id = tokenizer.pad_token_id


#tokenizer = GPT2Tokenizer.from_pretrained("/home/lihao/model/gpt2")
#model = GPT2LMHeadModel.from_pretrained("/home/lihao/model/gpt2")

print(model)

q = "我叫李灏，请你记住我的名字，我下面可能会问你的。"

q2 = """Translate English to French:

sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>"""

ids = tokenizer(q2, return_tensors='pt')["input_ids"].cuda()
#print(ids)
#print(type(ids))
final_outputs = model.generate(
    ids,
    do_sample=True,
    max_length=128,
    top_k=3,
    top_p=0.9,
    max_gen_len=64,
)

# print("final_outputs = {}, type(final_outputs) = {}, final_outputs.shape = {}".format(final_outputs, type(final_outputs), final_outputs.shape))

print("--result1: {}".format(tokenizer.decode(final_outputs[0], skip_special_tokens=True)))


q2 = "我叫什么名字?"
ids2 = tokenizer(q2, return_tensors='pt')["input_ids"].cuda()
#print(ids2)
#print(type(ids2))
final_outputs = model.generate(
    ids2,
    do_sample=True,
    #pad_token_id=model.config.eos_token_id,
    max_length=1024,
    top_k=3,
    top_p=0.95,
)

print("--result2: {}".format(tokenizer.decode(final_outputs[0], skip_special_tokens=True)))


pipeline = transformers.pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  torch_dtype=torch.float16,
  device_map="auto",
)

sequences = pipeline(
  q,
  do_sample=True,
  top_k=10,
  num_return_sequences=1,
  eos_token_id=tokenizer.eos_token_id,
  max_length=400,
)
for seq in sequences:
  print(f"{seq['generated_text']}")

"""
# qwen1 的输入
> /home/lihao/conda/miniconda3/envs/python311/lib/python3.11/site-packages/transformers/modeling_utils.py(3983)from_pretrained()
-> model.tie_weights()
(Pdb) c
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 4096)
    (layers): ModuleList(
      (0-31): 32 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((4096,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((4096,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((4096,), eps=1e-06)
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
"""


'''
# GPT2 模型
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
'''

"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
"""