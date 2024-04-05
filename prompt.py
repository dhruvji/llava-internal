from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
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
for img in image_paths[:100]: 
    image.append(Image.open(img).convert("RGB"))
for img in image_paths_red[:100]:
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
        token_idx=[-11, -6],
    )

json_file_path = f'/data/dhruv_gautam/llava_internal/caches/reg/activation_cache{timestamp}.json'
with open(json_file_path, 'w') as file:
    json.dump(activation_cache, file)

#print(activation_cache.shape)

inputs_red = processor([prompts[0] for i in image_red], images=image_red, padding=True, return_tensors="pt")
activation_cache_red = cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_red, 
        batch_size=1,
        token_idx=[-11, -6], 
    )

json_file_path = f'/data/dhruv_gautam/llava_internal/caches/red/activation_cache{timestamp}.json'
with open(json_file_path, 'w') as file:
    json.dump(activation_cache, file)

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