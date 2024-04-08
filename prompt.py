from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import pickle
import torch

import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

from utils import cache_activations, cache_activations_multimodal

def print_model_layers(model, indent=0):
    # Use .named_children() to get an iterator over immediate child modules, 
    # yielding both the name of the module as well as the module itself.
    for name, module in model.named_children():
        # Print module name with indentation for readability
        print(" " * indent + name + ": " + module.__class__.__name__)

        # If this module has further submodules, recursively print those
        if list(module.children()):
            print_model_layers(module, indent + 4)


save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

processor = AutoProcessor.from_pretrained(save_directory)
model = LlavaForConditionalGeneration.from_pretrained(save_directory)
n_layers = len(model.language_model.model.layers)
print(n_layers)
module_list = [f"model.language_model.model.layers[{i}]" for i in range(len(model.language_model.model.layers))]

processor.tokenizer.padding_side = "left"

image_dir = "/data/dhruv_gautam/coco/val2017/image/original"
image_dir_red = "/data/dhruv_gautam/coco/val2017/image/red_enhanced"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_red = [os.path.join(image_dir_red, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

prompts = [
        "USER: <image>\nPlease look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease add a red filter to the image, look at the image, and describe it\nASSISTANT:",
]
image = []
image_red = []
for img in image_paths[:10]: 
    image.append(Image.open(img).convert("RGB"))
for img in image_paths_red[:10]:
    image_red.append(Image.open(img).convert("RGB"))
input_text = "image, and describe it" #"\nASSISTANT:" is 5 tokens long
print(processor.tokenizer(input_text))
print("Processing prompts and image...")
inputs = processor([prompts[0] for i in image], images=image, padding=True, return_tensors="pt")
print(inputs)
print("Caching Activations...")
activation_cache = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs, 
        batch_size=1,
        token_idx=[-11], #-6
    )

pickle_file_path = f'/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_{timestamp}.pkl'
directory = os.path.dirname(pickle_file_path)
os.makedirs(directory, exist_ok=True)
with open(pickle_file_path, 'wb') as file:  
    pickle.dump(activation_cache, file)

#print(activation_cache.shape)

inputs_red = processor([prompts[0] for i in image_red], images=image_red, padding=True, return_tensors="pt")
activation_cache_red = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_red, 
        batch_size=1,
        token_idx=[-11], 
    )

pickle_file_path_red = f'/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_{timestamp}.pkl'
directory = os.path.dirname(pickle_file_path_red)
os.makedirs(directory, exist_ok=True)
with open(pickle_file_path_red, 'wb') as file:  
    pickle.dump(activation_cache_red, file)

inputs_prompt = processor([prompts[1] for i in image_red], images=image, padding=True, return_tensors="pt")
activation_cache_prompt = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_prompt, 
        batch_size=1,
        token_idx=[-11], 
    )

# activation_cache and activation_cache_red are lists of troch tensors
activation_cache_tensor = torch.stack(activation_cache)
activation_cache_flat = activation_cache_tensor.view(activation_cache_tensor.size(0), -1)
activation_cache_red_tensor = torch.stack(activation_cache_red)
activation_cache_red_flat = activation_cache_red_tensor.view(activation_cache_red_tensor.size(0), -1)
activation_cache_prompt_tensor = torch.stack(activation_cache_prompt)
activation_cache_prompt_flat = activation_cache_prompt_tensor.view(activation_cache_prompt_tensor.size(0), -1)

red_flat_direction = activation_cache_red_flat - activation_cache_flat  # apply this to a different flat activation_cache and see if cosine goes up with the current red

print(torch.nn.functional.normalize(red_flat_direction, p=2, dim=1))

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