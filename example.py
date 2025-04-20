# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

torch.cuda.set_device(1)  # 设置使用 GPU2（索引1）

# For reproducibility
random.seed(0)
torch.manual_seed(0)

# MODEL_NAME = 'Llama-3.1-8B-Instruct'
MODEL_NAME = 'Llama-2-7b-hf'

config = LlamaConfig.from_pretrained(f"/mnt/nvme4n1@164/tzj/models/{MODEL_NAME}")

config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
config.use_flash = True

model = LlamaForCausalLM_KIVI.from_pretrained(
    # pretrained_model_name_or_path='meta-llama/{MODEL_NAME}
    pretrained_model_name_or_path=f"/mnt/nvme4n1@164/tzj/models/{MODEL_NAME}",
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    f"/mnt/nvme4n1@164/tzj/models/{MODEL_NAME}", 
    use_fast=False, 
    trust_remote_code=True
)

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=256)
config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))

outputs = model(inputs, use_cache=True, output_attentions=True, output_hidden_states=True)
    
attentions = outputs.attentions
past_key_values = outputs.past_key_values
# past_key_values = torch.cat([outputs.past_key_values[1], outputs.past_key_values[5]], dim=0)

__import__('pdb').set_trace()

torch.save(past_key_values, f'./{MODEL_NAME}_kvcache.pt')
torch.save(attentions, f'./{MODEL_NAME}_attention.pt')

