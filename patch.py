from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

from utils import cache_activations_multimodal, generate_substitute_layer_single
prompts = [
        "USER: <image>\nPlease look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease add a red filter to the image, look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease look at the image and describe it and it's colors\nASSISTANT:",
]

def activation_patch(model, module_list, activation_cache_2, token_idx=-3, start_layer=20, end_layer=31):
    def forward_hook_factory(layer_idx, token_idx):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                new_output = output[0].clone()
            else:
                new_output = output.clone()
            if new_output.shape[1] > abs(token_idx):
                activation_2 = activation_cache_2[layer_idx][:, token_idx, :].clone()
                new_output[:, token_idx, :] = activation_2
            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            else:
                return new_output
        return forward_hook

    hook_handles = []
    for i, layer_str in enumerate(module_list[start_layer:end_layer + 1]):
        layer_module = eval(layer_str)
        hook = forward_hook_factory(i + start_layer, token_idx)
        handle = layer_module.register_forward_hook(hook)
        hook_handles.append(handle)

    return hook_handles

save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

processor = AutoProcessor.from_pretrained(save_directory)
model = LlavaForConditionalGeneration.from_pretrained(save_directory)
n_layers = len(model.language_model.model.layers)
print(n_layers)
module_list = [f"model.language_model.model.layers[{i}]" for i in range(len(model.language_model.model.layers))]

processor.tokenizer.padding_side = "left"

red = "/data/dhruv_gautam/random/images/red512.jpg"
black = "/data/dhruv_gautam/random/images/black512.jpg"
image_dir = "/data/dhruv_gautam/coco/val2017/image/original"
image_dir_red = "/data/dhruv_gautam/coco/val2017/image/red_enhanced"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_red = [os.path.join(image_dir_red, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

image = []
image_red = []
image.append(Image.open(black).convert('RGB'))
image_red.append(Image.open(red).convert('RGB'))

inputs = processor([prompts[0] for i in image], images=image, padding=True, return_tensors="pt")
print("Caching Activations...")
activation_cache_black = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs, 
        batch_size=1,
        token_idx=[-4, -3, -2, -1],
    )

inputs_red = processor([prompts[0] for i in image_red], images=image_red, padding=True, return_tensors="pt")
activation_cache_red = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_red, 
        batch_size=1,
        token_idx=[-4, -3, -2, -1], 
    )
"""
print("Generating GT Output...")
output = model.generate(**inputs, max_new_tokens=20)
print("Decoding GT output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
print("Generating GT Output...")
output = model.generate(**inputs_red, max_new_tokens=20)
print("Decoding GT output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
"""
hook_handles = activation_patch(model, module_list, activation_cache_red, token_idx=-3, start_layer=20, end_layer=31)

print("Generating Patch Red Output...")
output = model.generate(**inputs, max_new_tokens=20)
print("Decoding Patch Red output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
print("Generating Patch Red Output...")
output = model.generate(**inputs_red, max_new_tokens=20)
print("Decoding Patch Red output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)

for handle in hook_handles:
    handle.remove()

hook_handles = activation_patch(model, module_list, activation_cache_black, token_idx=-3, start_layer=20, end_layer=31)

print("Generating Patch Black Output...")
output = model.generate(**inputs, max_new_tokens=20)
print("Decoding Patch Black output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
print("Generating Patch Black Output...")
output = model.generate(**inputs_red, max_new_tokens=20)
print("Decoding Patch Black output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)

for handle in hook_handles:
    handle.remove()