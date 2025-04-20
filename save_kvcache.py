# LLaMA FP16版本保存KV Cache
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer  # 使用原生Llama模型
from datasets import load_dataset

torch.cuda.set_device(0)  # 设置使用 GPU2（索引1）

# For reproducibility
random.seed(0)
torch.manual_seed(0)

# MODEL_NAME = 'Llama-3.1-8B-Instruct'
MODEL_NAME = 'Llama-2-13b-hf'
# MODEL_NAME = 'Llama-2-7b-hf'

# 加载原生配置（移除所有量化相关参数）
config = LlamaConfig.from_pretrained(f"/mnt/nvme4n1@164/tzj/models/{MODEL_NAME}")
config.use_cache = True  # 确保启用缓存

# 加载原生FP16模型（不使用KIVI修改）
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=f"/mnt/nvme4n1@164/tzj/models/{MODEL_NAME}",
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,  # FP16精度
).cuda()

enc = AutoTokenizer.from_pretrained(
    f"/mnt/nvme4n1@164/tzj/models/{MODEL_NAME}", 
    use_fast=False, 
    trust_remote_code=True
)

# 生成阶段（保持相同输入）
dataset = load_dataset('gsm8k', 'main')
prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

# 生成并保存KV Cache
with torch.inference_mode():
    # 前向传播获取缓存
    outputs = model(
        input_ids=inputs,
        use_cache=True,
        output_attentions=True,   # 需要注意力权重则保留
        output_hidden_states=True # 需要隐藏状态则保留
    )
    
    # 提取并保存FP16格式的KV Cache
    past_key_values = outputs.past_key_values
    torch.save(past_key_values, f'./{MODEL_NAME}_fp16_kvcache.pt')
    
    # 可选：保存注意力权重
    if outputs.attentions is not None:
        attentions = outputs.attentions
        torch.save(attentions, f'./{MODEL_NAME}_fp16_attention.pt')

# 验证生成结果（可选）
output = model.generate(inputs, max_new_tokens=96)
print("\nGenerated Text:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))