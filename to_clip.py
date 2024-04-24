import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch.nn as nn
from torch.nn import InstanceNorm1d
import torch.optim as optim
import pickle
import os
import datetime
import random
import torch.nn.functional as F
timestamp = datetime.datetime.now().strftime("%d%H%M")

device = torch.device("cuda")

from utils import batched_cache_activations_multimodal

save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

processor = AutoProcessor.from_pretrained(save_directory)
model = LlavaForConditionalGeneration.from_pretrained(save_directory)
n_layers = len(model.language_model.model.layers)
print(n_layers)
module_list = [f"model.language_model.model.layers[{i}]" for i in range(len(model.language_model.model.layers))]

processor.tokenizer.padding_side = "left"

image_dir = "/data/dhruv_gautam/coco/val2017/image_grey/original"
image_dir_grey = "/data/dhruv_gautam/coco/val2017/image_grey/greyscale"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
image_paths_grey = [os.path.join(image_dir_grey, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

prompts = [
        "USER: <image>\nPlease look the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease add a red filter to the image, look at the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease look at the image and describe it and it's colors\nASSISTANT:",
        "USER: <image>\nPlease visualize the image and describe it\nASSISTANT:",
        "USER: <image>\nPlease visualize the image with a stronger red channel and describe it\nASSISTANT:",
        "USER: <image>\nPlease imagine the image with a stronger red channel and describe it\nASSISTANT:",
        "USER: <image>\nPlease visualize the image colorized and describe it\nASSISTANT:",
]
image = []
image_grey = []

for img in image_paths[:500]: 
    image.append(Image.open(img).convert("RGB"))
for img in image_paths_grey[:500]:
    image_grey.append(Image.open(img).convert("RGB"))

print("Processing prompts and image...")
inputs = processor([prompts[0] for _ in image], images=image, padding=True, return_tensors="pt")
print(len(inputs))
print("Caching Activations normal...")

activation_cache_normal = batched_cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs, 
        batch_size=40,
        token_idx=[-3],  # 4 5 6 7 is the image itself
    )
print(len(activation_cache_normal))
print(len(activation_cache_normal[0]))
print(len(activation_cache_normal[0][0]))
print(len(activation_cache_normal[0][0][0]))
"""
inputs_grey = processor([prompts[6] for _ in image_grey], images=image_grey, padding=True, return_tensors="pt")
print("Caching Activations grey...")

activation_cache_grey = batched_cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,  
        cache_input_output='output',
        inputs=inputs_grey, 
        batch_size=40,
        token_idx=[-3], 
    )
"""

class MultiLayerEmbeddingProbe(nn.Module):
    def __init__(self, input_features, output_size, hidden_size=1024):
        super(MultiLayerEmbeddingProbe, self).__init__()
        self.linear1 = nn.Linear(input_features, hidden_size)
        self.norm1 = InstanceNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = InstanceNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu1(self.norm1(self.linear1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.norm2(self.linear2(x)))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

embedding_size = 768
num_epochs = 300
layer_start = 0
layer_end = 31

color_clip_embeddings_path = f'/data/dhruv_gautam/llava-internal/clip/original_embeddings222044.pth'

clip_embeddings = torch.load(color_clip_embeddings_path)
clip_embeddings = [batch.to(device) for batch in clip_embeddings]
clip_embeddings_train = clip_embeddings[:90] #batched for 40 this means 200
clip_embeddings_test = clip_embeddings[90:] # old 90
print(len(clip_embeddings))

input_features = sum(activation_cache_normal[j][0][k].numel() for j in range(layer_start, layer_end + 1) for k in range(1))

probe = MultiLayerEmbeddingProbe(input_features, embedding_size).to(device)
criterion = nn.MSELoss()
#criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(probe.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()
    batch_index = 0
    for batch_embeddings in clip_embeddings_train: 
        for i in range(batch_embeddings.size(0)):  
            if batch_index + i >= len(activation_cache_normal[layer_start]):
                break
            primary_normal = torch.cat([
                activation_cache_normal[j][batch_index + i][k].view(-1)
                for k in range(1)  
                for j in range(layer_start, layer_end + 1)  
            ]).unsqueeze(0).to(device)            
            output = probe(primary_normal)
            target_embedding = batch_embeddings[i].unsqueeze(0)  
            #target_labels = torch.tensor([1], device=device)  
            
            if target_embedding.size(1) != output.size(1):
                print("Dimension mismatch.")
                continue
            
            loss = criterion(output, target_embedding)
            #loss = criterion(output, target_embedding, target_labels)
            loss.backward()
            total_loss += loss.item()

        batch_index += batch_embeddings.size(0)
        
    optimizer.step()
    avg_loss = total_loss / (batch_index)
    scheduler.step(avg_loss)
    print(f'Epoch {epoch + 1}, Average Loss: {total_loss / (batch_index)}')
    

print('Finished Training Probe')

os.makedirs(f'/data/dhruv_gautam/llava-internal/probes/', exist_ok=True)
model_save_path = f'/data/dhruv_gautam/llava-internal/probes/ist_clip_{timestamp}.pth'
torch.save(probe.state_dict(), model_save_path)
print("Saved trained probe.")

probe = MultiLayerEmbeddingProbe(input_features, embedding_size).to(device)
probe.load_state_dict(torch.load(model_save_path))

batch_index = 0
for batch_embeddings in clip_embeddings_test:
    for i in range(batch_embeddings.size(0)): 
        if 450 + batch_index + i >= len(activation_cache_normal[layer_start]):
            print(len(activation_cache_normal[layer_start]))
            print("out of bounds")
            break
        primary_normal = torch.cat([
            activation_cache_normal[j][batch_index + i][k].view(-1)
            for k in range(1)  
            for j in range(layer_start, layer_end + 1)  
        ]).unsqueeze(0).to(device)          
        output = probe(primary_normal)
        target_embedding = batch_embeddings[i].unsqueeze(0)
        sim = F.cosine_similarity(output, target_embedding.unsqueeze(0)).mean().item()
        print(sim)
    batch_index += batch_embeddings.size(0)
