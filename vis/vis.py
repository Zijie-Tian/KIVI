import os
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.path.dirname(os.path.abspath(__file__))
FONTSIZE = 16
# MODEL_NAME = 'Llama-3.1-8B-Instruct'
MODEL_NAME = 'Llama-2-7b-hf'

font_config = {'font.size': FONTSIZE, 'font.family': 'DejaVu Math TeX Gyre'}
plt.rcParams.update(font_config)
plt.rcParams["figure.figsize"] = (4, 4.5)

# Specify the output folder
output_folder = os.path.join(current_dir, MODEL_NAME)
os.makedirs(output_folder, exist_ok=True)

project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# kv_filename = os.path.join(project_dir, f'{MODEL_NAME}_fp16_kvcache.pt')
# attn_filename = os.path.join(project_dir, f'{MODEL_NAME}_fp16_attention.pt')
attn_filename = os.path.join(project_dir, f'{MODEL_NAME}_attention.pt')
kv_filename = os.path.join(project_dir, f'{MODEL_NAME}_kvcache.pt')
kvcache = torch.load(kv_filename, map_location='cpu', weights_only=False)
attentions = torch.load(attn_filename, map_location='cpu', weights_only=False)

for layer_id in [3, 8, 14, 16, 18, 20, 31]:  # Replace with your layer ids
    head_id = 0
    k, v = kvcache[layer_id][0].squeeze(0), kvcache[layer_id][1].squeeze(0)
    # k = torch.sort(k, dim=-1)[0]
    # v = torch.sort(v, dim=-1)[0]

    # k, v = kvcache[layer_id][1].squeeze(0), kvcache[layer_id][5].squeeze(0)
    k = k.transpose(0, 1).abs().detach().numpy()
    v = v.transpose(0, 1).abs().detach().numpy()
    k, v = k[:, head_id, :], v[:, head_id, :]

    # Iterate over key and value tensors
    for idx, tensor in enumerate([k, v]):
        tokens, channels = tensor.shape
        x = np.arange(tokens)
        y = np.arange(channels)
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, tensor.T, cmap='coolwarm')
        
        ax.xaxis.set_tick_params(pad=-5)
        ax.yaxis.set_tick_params(pad=-3)
        ax.zaxis.set_tick_params(pad=-130)

        ax.set_xlabel('Token', labelpad=-5)
        ax.set_ylabel('Column', labelpad=-1)
        if layer_id in [3, 16]:
            ax.zaxis.set_rotate_label(False)
        
        if idx == 0:
            filename = f'{MODEL_NAME}_layer{layer_id}_head{head_id}_k.pdf'
        else:
            filename = f'{MODEL_NAME}_layer{layer_id}_head{head_id}_v.pdf'
        
        save_filename = os.path.join(output_folder, filename)
        plt.savefig(save_filename, bbox_inches='tight')
        plt.close(fig)
