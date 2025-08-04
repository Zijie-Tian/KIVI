# LLaMA model with KIVI
import warnings

warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from quant.new_pack import (
    quant_and_pack_vcache,
    unpack_and_dequant_kcache,
    triton_quantize_and_pack_along_last_dim,
    unpack_and_dequant_vcache,
    quant_and_pack_kcache,
)
from datasets import load_dataset

import numpy as np

np.set_printoptions(precision=4, suppress=True)

torch.cuda.set_device(1)  # 设置使用 GPU2（索引1）

# For reproducibility
random.seed(0)
torch.manual_seed(0)

# MODEL_NAME = 'Llama-3.1-8B-Instruct'
MODEL_NAME = "Llama-2-7b-hf"

config = LlamaConfig.from_pretrained(f"/workspace/models/{MODEL_NAME}")

config.k_bits = 2  # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32
config.residual_length = 32  # corresponding to the number of recent fp16 tokens
config.use_flash = True
# TODO: USE following config to evaluate smoothing mechianism
config.smooth_step = 8

model = LlamaForCausalLM_KIVI.from_pretrained(
    # pretrained_model_name_or_path='meta-llama/{MODEL_NAME}
    pretrained_model_name_or_path=f"/workspace/models/{MODEL_NAME}",
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    f"/workspace/models/{MODEL_NAME}", use_fast=False, trust_remote_code=True
)

dataset = load_dataset("/workspace/data/gsm8k_local", "main")

prompt = ""
for i in range(5):
    prompt += (
        "Question: "
        + dataset["train"][i]["question"]
        + "\nAnswer: "
        + dataset["train"][i]["answer"]
        + "\n"
    )
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=256)
config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

print(prompt + "\n" + "=" * 10 + f"\n{config_str}\n" + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1] :], skip_special_tokens=True))

# NOTICE: we must use `use_cache=True` to get the past_key_values
outputs = model(
    inputs, use_cache=True, output_attentions=True, output_hidden_states=True
)

attentions = outputs.attentions
past_key_values = outputs.past_key_values

__import__("pdb").set_trace()

kvcache = []
for layer_id in range(len(past_key_values)):
    key_states_quant_trans = past_key_values[layer_id][0]  # qval.T
    key_states_full = past_key_values[layer_id][1]
    key_scale_trans = past_key_values[layer_id][2]  # scale.T
    key_mn_trans = past_key_values[layer_id][3]  # zero_point.T
    value_states_quant = past_key_values[layer_id][4]  # qval.
    value_states_full = past_key_values[layer_id][5]
    value_scale = past_key_values[layer_id][6]  # scale
    value_mn = past_key_values[layer_id][7]  # zero_point
    # NOTE: Add `smooth varible.`
    key_states_smooth = past_key_values[layer_id][8]  # key_states_smooth
    value_states_smooth = past_key_values[layer_id][9]  # value_states_smooth

    # NOTE: Dequantized  key and value.
    dequant_key = unpack_and_dequant_kcache(
        key_states_quant_trans,
        key_scale_trans.unsqueeze(-1),
        key_mn_trans.unsqueeze(-1),
        config.group_size,
        config.k_bits,
    )

    # NOTE: Subtract the smooth variable.
    batch, num_heads, seq_len, dim = dequant_key.shape
    dequant_key = dequant_key.view(
        batch, num_heads, -1, config.group_size, dim
    ) - key_states_smooth.unsqueeze(3).expand(-1, -1, -1, config.group_size, -1)
    dequant_key = dequant_key.view(batch, num_heads, seq_len, dim)

    dequant_value = unpack_and_dequant_vcache(
        value_states_quant,
        value_scale.unsqueeze(-1),
        value_mn.unsqueeze(-1),
        config.group_size,
        config.v_bits,
    )
    kvcache.append([dequant_key, dequant_value])

__import__("pdb").set_trace()

# torch.save(kvcache, f'./{MODEL_NAME}_kvcache.pt')
# torch.save(attentions, f'./{MODEL_NAME}_attention.pt')
