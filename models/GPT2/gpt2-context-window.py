from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("/home/lihao/model/gpt2")
model = GPT2LMHeadModel.from_pretrained("/home/lihao/model/gpt2")
# print(type(model))

# 单请求
q = "My name is lihao, please record my name."
q2 = "what's my name?"

ids = tokenizer.encode(q, return_tensors='pt')
#print(ids)
#print(type(ids))
final_outputs = model.generate(
    ids,
    do_sample=True,
    max_length=100,
    pad_token_id=model.config.eos_token_id,
    top_k=3,
    top_p=0.95,
)

# print("final_outputs = {}, type(final_outputs) = {}, final_outputs.shape = {}".format(final_outputs, type(final_outputs), final_outputs.shape))

print("--result1: {}".format(tokenizer.decode(final_outputs[0], skip_special_tokens=True)))



q2 = "what's my name?"
ids2 = tokenizer.encode(q2, return_tensors='pt')
#print(ids2)
#print(type(ids2))
final_outputs = model.generate(
    ids2,
    do_sample=True,
    max_length=100,
    pad_token_id=model.config.eos_token_id,
    top_k=3,
    top_p=0.95,
)

print("--result2: {}".format(tokenizer.decode(final_outputs[0], skip_special_tokens=True)))