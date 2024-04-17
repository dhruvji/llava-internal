import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

image_dir = "/data/dhruv_gautam/coco/val2017/image_grey/original"
image_dir_grey = "/data/dhruv_gautam/coco/val2017/image_grey/greyscale"

image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_grey = [os.path.join(image_dir_grey, filename) for filename in os.listdir(image_dir_grey) if filename.endswith('.jpg')]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_clip_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs

batch_size = 5  
embeddings = []
for i in range(0, len(image_paths[:250]), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_embeddings = torch.cat([get_clip_embedding(img_path) for img_path in batch_paths], dim=0)
    embeddings.append(batch_embeddings)
    print(f"batch {i}")

os.makedirs(f"/data/dhruv_gautam/llava-internal/clip/", exist_ok=True)
torch.save(embeddings, f'/data/dhruv_gautam/llava-internal/clip/original_embeddings{timestamp}.pth')

embeddings_grey = []
for i in range(0, len(image_paths_grey[:250]), batch_size):
    batch_paths_grey = image_paths_grey[i:i+batch_size]
    batch_embeddings_grey = torch.cat([get_clip_embedding(img_path) for img_path in batch_paths_grey], dim=0)
    embeddings_grey.append(batch_embeddings_grey)
    print(f"batch {i}")

torch.save(embeddings_grey, f'/data/dhruv_gautam/llava-internal/clip/grey_embeddings{timestamp}.pth')

print("Embeddings generated and saved.")
