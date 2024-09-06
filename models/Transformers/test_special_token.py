from transformers import LlamaTokenizer

# 加载 Llama-2 的 tokenizer
tokenizer = LlamaTokenizer.from_pretrained("/home/lihao/model/meta-llama/Llama-2-7b-hf")

# 获取特殊字符的数量
special_tokens = tokenizer.special_tokens_map
num_special_tokens = len(special_tokens)
"""
特殊字符: {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
特殊字符数量: 3
"""
print("特殊字符:", special_tokens)
print("特殊字符数量:", num_special_tokens)


# 示例输入
q2 = [
    """中国有几个省？有上海、江苏""",
    "我今晚要去吃火锅，有牛肉卷，有"
]
tokenizer.pad_token = tokenizer.unk_token

# 很重要，
"""
{'input_ids': tensor([[    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             1, 29871, 30275, 30356, 30417,   232,   138,   163, 30502, 31600,
         30882, 30417, 30429, 30581, 30330, 30775,   235,   142,   146],
        [    1, 29871, 30672, 31482,   233,   156,   157, 30698, 31475,   232,
           147,   134, 31313,   236,   151,   136, 30214, 30417,   234,   140,
           158,   235,   133,   140,   232,   144,   186, 30214, 30417]]), 'attention_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1]])}
"""
tokenizer.padding_side = "left"
# 编码输入并自动填充
encoded_inputs = tokenizer(q2, padding=True, truncation=True, return_tensors="pt")

print(encoded_inputs)

# llama2 的分词器问题
# https://blog.csdn.net/weixin_41496173/article/details/140062526