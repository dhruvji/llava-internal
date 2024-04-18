from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

from utils import cache_activations_multimodal, batched_cache_activations_multimodal

def print_model_layers(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent + name + ": " + module.__class__.__name__)
        if list(module.children()):
            print_model_layers(module, indent + 4)


save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

processor = AutoProcessor.from_pretrained(save_directory)
model = LlavaForConditionalGeneration.from_pretrained(save_directory)
n_layers = len(model.language_model.model.layers)
print(n_layers)
module_list = [f"model.language_model.model.layers[{i}]" for i in range(len(model.language_model.model.layers))]

processor.tokenizer.padding_side = "left"

red = "/data/dhruv_gautam/random/images/red512.jpg"
black = "/data/dhruv_gautam/random/images/black512.jpg"
image_dir = "/data/dhruv_gautam/coco/val2017/image_grey/original"
image_dir_red = "/data/dhruv_gautam/coco/val2017/image_grey/greyscale"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_red = [os.path.join(image_dir_red, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

prompts = [
        "USER: <image>\nPlease look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease add a red filter to the image, look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease look at the image and describe it and it's colors\nASSISTANT:",
        "USER: <image>\nPlease visualize the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease visualize the image with a stronger red channel and describe it\nASSISTANT:",
        "USER: <image>\nPlease imagine the image with a stronger red channel and describe it\nASSISTANT:",
        "USER: <image>\nPlease visualize the image colorized and describe it\nASSISTANT:",
]
image = []
image_red = []
"""
image.append(Image.open(black).convert('RGB'))
image_red.append(Image.open(red).convert('RGB'))
"""
for img in image_paths[:250]: 
    image.append(Image.open(img).convert("RGB"))
for img in image_paths_red[:250]:
    image_red.append(Image.open(img).convert("RGB"))

input_text = "USER: <image>\nPlease look at the image and describe it\nASSISTANT:" #"\nASSISTANT:" is 5 tokens long
print(processor.tokenizer(input_text))
input_text = "USER: <image>\n" #"\nASSISTANT:" is 5 tokens long
print(processor.tokenizer(input_text))
input_text = "Please look at the image and describe it\nASSISTANT:" #"\nASSISTANT:" is 5 tokens long
print(processor.tokenizer(input_text))
print("Processing prompts and image...")
inputs = processor([prompts[6] for i in image], images=image, padding=True, return_tensors="pt")
print(len(inputs))
print("Caching Activations...")
"""
activation_cache = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs, 
        batch_size=20,
        #token_idx=[-3], #-14, -13, -12, -11, -10, -9, 
        token_idx=[-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1], 
    )
"""
activation_cache = batched_cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs, 
        batch_size=40,
        #token_idx=[-3], #-14, -13, -12, -11, -10, -9, 
        token_idx=[-23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1], 
    )

pickle_file_path = f'/data/dhruv_gautam/llava-internal/caches/reg/grey_main_cache{timestamp}.pkl'
#pickle_file_path = f'/data/dhruv_gautam/llava-internal/caches/reg/describe_test_cache{timestamp}.pkl'
directory = os.path.dirname(pickle_file_path)
os.makedirs(directory, exist_ok=True)
with open(pickle_file_path, 'wb') as file:  
    pickle.dump(activation_cache, file)
    print("saved normal to cache")
print(len(activation_cache))
print(len(activation_cache[0]))
print(len(activation_cache[0][0]))
print(len(activation_cache[0][0][0]))

#print(activation_cache.shape)

inputs_red = processor([prompts[6] for i in image_red], images=image_red, padding=True, return_tensors="pt")
"""
activation_cache_red = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_red, 
        batch_size=1,
        #token_idx=[-3],
        token_idx=[-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1], 
    )
""" 
activation_cache_red = batched_cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_red, 
        batch_size=40,
        #token_idx=[-3],
        token_idx=[-23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1], 
    )

pickle_file_path_red = f'/data/dhruv_gautam/llava-internal/caches/red/grey_main_cache{timestamp}.pkl'
#pickle_file_path_red = f'/data/dhruv_gautam/llava-internal/caches/red/describe_test_cache{timestamp}.pkl'
directory = os.path.dirname(pickle_file_path_red)
os.makedirs(directory, exist_ok=True)
with open(pickle_file_path_red, 'wb') as file:  
    pickle.dump(activation_cache_red, file)
    print("saved red to cache")
"""
inputs_prompt = processor([prompts[1] for i in image_red], images=image, padding=True, return_tensors="pt")
activation_cache_prompt = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_prompt, 
        batch_size=1,
        token_idx=[-14, -13, -12, -11, -10, -9, -8, -7, -6], #
    )
"""

"""
# activation_cache are lists of troch tensors by the layers of the model (32)
for x in range(len(activation_cache)):
    print(f"\n======Layer {x}=======")
    #activation_cache_tensor = torch.stack(activation_cache[x])
    activation_cache_flat = activation_cache[x].view(activation_cache[x].size(0), -1)
    #activation_cache_red_tensor = torch.stack(activation_cache_red[x])
    activation_cache_red_flat = activation_cache_red[x].view(activation_cache_red[x].size(0), -1)
    #activation_cache_prompt_tensor = torch.stack(activation_cache_prompt[x])
    activation_cache_prompt_flat = activation_cache_prompt[x].view(activation_cache_prompt[x].size(0), -1)

    #red_flat_direction = activation_cache_red_flat - activation_cache_flat  # apply this to a different flat activation_cache and see if cosine goes up with the current red

    #print(torch.nn.functional.normalize(red_flat_direction, p=2, dim=1))

    activation_cache_norm = torch.nn.functional.normalize(activation_cache_flat, p=2, dim=1)
    activation_cache_red_norm = torch.nn.functional.normalize(activation_cache_red_flat, p=2, dim=1)
    activation_cache_prompt_norm = torch.nn.functional.normalize(activation_cache_prompt_flat, p=2, dim=1)
    cosine_sim = torch.mm(activation_cache_norm, activation_cache_red_norm.transpose(0, 1))
    print("\nCosine sim normal and red")
    print(cosine_sim)
    cosine_sim_p = torch.mm(activation_cache_norm, activation_cache_prompt_norm.transpose(0, 1))
    print("\nCosine sim normal and red prompt")
    print(cosine_sim_p)
    cosine_sim_rp = torch.mm(activation_cache_red_norm, activation_cache_prompt_norm.transpose(0, 1))
    print("\nCosine sim red and red prompt")
    print(cosine_sim_rp)
"""
instance_index = 1
token_positions = [-23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1] # -17
token_ids = inputs["input_ids"][0]  
tokens_all = [processor.tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
tokens_specific = [tokens_all[idx] for idx in token_positions]
for layer_index in range(len(model.language_model.model.layers)):
    print(layer_index)
    activation_instance_normal = activation_cache[layer_index][instance_index]  
    activation_instance_red = activation_cache_red[layer_index][instance_index] 

    cosine_sim_matrix = np.zeros((activation_instance_normal.size(0), activation_instance_red.size(0)))

    for i in range(activation_instance_normal.size(0)): 
        for j in range(activation_instance_red.size(0)):  
            token_i_norm = torch.nn.functional.normalize(activation_instance_normal[i].unsqueeze(0), p=2, dim=1)
            token_j_norm = torch.nn.functional.normalize(activation_instance_red[j].unsqueeze(0), p=2, dim=1)        
            cosine_sim = torch.mm(token_i_norm, token_j_norm.transpose(0, 1)).item()
            cosine_sim_matrix[i, j] = cosine_sim

    plt.figure(figsize=(8, 6))
    plt.imshow(cosine_sim_matrix, cmap='viridis', interpolation='nearest')
    
    plt.xticks(np.arange(len(tokens_specific)), tokens_specific, rotation='vertical')
    plt.yticks(np.arange(len(tokens_specific)), tokens_specific)
    plt.title(f'Cosine Similarity Across Tokens: Layer {layer_index}, Instance {instance_index}')
    plt.colorbar()
    plt.xlabel('Token Positions in Red Cache')
    plt.ylabel('Token Positions in Black Cache')
    plt.xticks(range(activation_instance_normal.size(0)))
    plt.yticks(range(activation_instance_red.size(0)))
    #os.makedirs(f"/data/dhruv_gautam/figures/describe_test_cache{timestamp}/", exist_ok=True)
    #plt.savefig(f"/data/dhruv_gautam/figures/describe_test_cache{timestamp}/layer{layer_index}.jpg")
    os.makedirs(f"/data/dhruv_gautam/figures/grey_main{timestamp}/", exist_ok=True)
    plt.savefig(f"/data/dhruv_gautam/figures/grey_main{timestamp}/layer{layer_index}.jpg")

"""
num_layers = len(model.language_model.model.layers)
num_instances = len(activation_cache[0])  
diagonal_cosine_sims = np.zeros((num_layers, num_instances))

for instance_index in range(num_instances):
    token_ids = inputs["input_ids"][instance_index]  
    tokens_all = [processor.tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
    token_positions = [-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
    tokens_specific = [tokens_all[idx] for idx in token_positions]

    for layer_index in range(num_layers):
        activation_instance_normal = activation_cache[layer_index][instance_index]  
        activation_instance_red = activation_cache_red[layer_index][instance_index] 

        cosine_sim_matrix = np.zeros((len(tokens_specific), len(tokens_specific)))

        for i in range(len(tokens_specific)):
            token_i_norm = torch.nn.functional.normalize(activation_instance_normal[i].unsqueeze(0), p=2, dim=1)
            token_j_norm = torch.nn.functional.normalize(activation_instance_red[i].unsqueeze(0), p=2, dim=1) 
            cosine_sim = torch.mm(token_i_norm, token_j_norm.transpose(0, 1)).item()
            cosine_sim_matrix[i, i] = cosine_sim
        
        diagonal_cosine_sims[layer_index, instance_index] = np.mean(np.diag(cosine_sim_matrix))

average_diagonal_cosine_sims = np.mean(diagonal_cosine_sims, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(num_layers), average_diagonal_cosine_sims, marker='o')
plt.title('Average Diagonal Cosine Similarity per Layer')
plt.xlabel('Layer Index')
plt.ylabel('Average Diagonal Cosine Similarity')
plt.grid(True)
plt.savefig(f"/data/dhruv_gautam/figures/average_cosine_similarity{timestamp}.jpg")
plt.show()
"""

#print(activation_cache.shape)
print("Generating Output...")
output = model.generate(**inputs, max_new_tokens=20)
print("Decoding output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
print("Generating Output...")
output = model.generate(**inputs_red, max_new_tokens=20)
print("Decoding output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)