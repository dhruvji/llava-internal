from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image
import os

save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

processor = AutoProcessor.from_pretrained(save_directory)
model = AutoModelForPreTraining.from_pretrained(save_directory)

image_dir = "/data/dhruv_gautam/llava-internal/stl10_images/original"
image_dir_red = "/data/dhruv_gautam/llava-internal/stl10_images/red_enhanced"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_red = [os.path.join(image_dir_red, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

prompts = [
        "USER: <image>\nPlease describe this image\nASSISTANT:",
]

image = Image.open(image_paths[0]).convert("RGB")
inputs = processor(prompts, images=[image], padding=True, return_tensors="pt")
print("Inputs created")

output = model.generate(**inputs, max_new_tokens=20)
print("Output Generated")

generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)