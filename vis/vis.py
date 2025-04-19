import torch
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

FONTSIZE = 16

font_config = {'font.size': FONTSIZE, 'font.family': 'DejaVu Math TeX Gyre'}
plt.rcParams.update(font_config)
plt.rcParams["figure.figsize"] = (4, 4.5)

# generate kv cache and attention
# inputs = enc(sample, return_tensors='pt').to('cuda')
# outputs = model(inputs['input_ids'], use_cache=True, output_attentions=True)
# past_key_values = outputs.past_key_values
# attentions = outputs.attentions
# torch.save(past_key_values, f'./{model}_kvcache.pt')
# torch.save(attentions, f'./{model}_attention.pt')

MODEL_NAME = 'Llama-3.1-8B-Instruct'

model = 'Llama-2-7b-hf' # replace with your model name
kv_filename = f'./{MODEL_NAME}_kvcache.pt'
attn_filename = f'./{MODEL_NAME}_attention.pt'
kvcache = torch.load(kv_filename, map_location='cpu')
attentions = torch.load(attn_filename, map_location='cpu')

for layer_id in [3, 8, 14, 16, 18, 20, 31]: # replace with your layer ids
    head_id = 0
    k, v = kvcache[layer_id][0].squeeze(0), kvcache[layer_id][1].squeeze(0)

    k = k.transpose(0, 1).abs().detach().numpy()
    v = v.transpose(0, 1).abs().detach().numpy()
    k, v = k[:, head_id, :], v[:, head_id, :]

    # Sample 2D tensor (replace this with your actual tensor)
    for idx, tensor in enumerate([k, v]):
        # Creating a meshgrid
        tokens, channels = tensor.shape
        x = np.arange(tokens)
        y = np.arange(channels)
        X, Y = np.meshgrid(x, y)
        # Creating a figure and a 3D subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotting the surface
        surf = ax.plot_surface(X, Y, tensor.T, cmap='coolwarm')

        ax.xaxis.set_tick_params(pad=-5)
        ax.yaxis.set_tick_params(pad=-3)
        ax.zaxis.set_tick_params(pad=-130)

        # Adding labels
        ax.set_xlabel('Token', labelpad=-5)
        ax.set_ylabel('Column', labelpad=-1)
        if layer_id in [3, 16]:
            ax.zaxis.set_rotate_label(False) 
        if idx == 0:
            save_filename = f'./{model}_layer{layer_id}_head{head_id}_k.pdf'
        else:
            save_filename = f'./{model}_layer{layer_id}_head{head_id}_v.pdf'
        plt.savefig(save_filename, bbox_inches='tight')
