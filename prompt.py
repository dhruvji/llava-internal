from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import torch

from utils import cache_activations

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

image_dir = "/data/dhruv_gautam/flickr30k_images/flickr30k_images/tempdir/image/original"
image_dir_red = "/data/dhruv_gautam/flickr30k_images/flickr30k_images/tempdir/image/red_enhanced"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_red = [os.path.join(image_dir_red, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

prompts = [
        "USER: <image>\nPlease look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease add a red filter to the image, look at the image, and describe it\nASSISTANT:",
]

image = Image.open(image_paths[0]).convert("RGB")
image_red = Image.open(image_paths_red[0]).convert("RGB")
input_text = "USER: <image>\nPlease look at the image and describe it's colors\nASSISTANT:" #\nASSISTANT: is 5 tokens long
print(processor.tokenizer(input_text))
print("Processing prompts and image...")
inputs = processor(prompts, images=[image, image_red], padding=True, return_tensors="pt")
print("Caching Activations...")
activation_cache = cache_activations(
        model=model,
        tokenizer=processor.tokenizer,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs, 
        batch_size=50,
        token_idx=[-6],
    )
print("Generating Output...")
output = model.generate(**inputs, max_new_tokens=20)
print("Decoding output...")
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)