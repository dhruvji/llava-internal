import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import datetime
import random
import torch.nn.functional as F
timestamp = datetime.datetime.now().strftime("%d%H%M")

device = torch.device("cuda")

class MultiLayerEmbeddingProbe(nn.Module):
    def __init__(self, input_features, output_size):
        super(MultiLayerEmbeddingProbe, self).__init__()
        self.linear = nn.Linear(input_features, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x))
        return x

embedding_size = 768  
num_epochs = 200
token_index = -3
layer_start = 0
layer_end = 30

pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/grey_main_cache180344.pkl'
pickle_file_path_grey = '/data/dhruv_gautam/llava-internal/caches/red/grey_main_cache180344.pkl'
color_clip_embeddings_path = f'/data/dhruv_gautam/llava-internal/clip/original_embeddings170346.pth'

with open(pickle_file_path_normal, 'rb') as file:
    activation_cache_normal = pickle.load(file)
print(f"Entries per layer: {len(activation_cache_normal[0])}")
print(f"Entries per layer: {len(activation_cache_normal[0][0])}")

with open(pickle_file_path_grey, 'rb') as file:
    activation_cache_grey = pickle.load(file)

clip_embeddings = torch.load(color_clip_embeddings_path)
clip_embeddings = [batch.to(device) for batch in clip_embeddings]
clip_embeddings_train = clip_embeddings[:40]
clip_embeddings_test = clip_embeddings[40:]
print(len(clip_embeddings))

input_features = sum(activation_cache_normal[i][0][token_index].numel() for i in range(layer_start, layer_end + 1))

probe = MultiLayerEmbeddingProbe(input_features, embedding_size).to(device)
criterion = nn.CosineEmbeddingLoss()
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
            primary_normal = torch.cat([activation_cache_normal[j][batch_index + i][token_index].view(-1) for j in range(layer_start, layer_end + 1)]).unsqueeze(0).to(device)
            output = probe(primary_normal)
            target_embedding = batch_embeddings[i].unsqueeze(0)  
            target_labels = torch.tensor([1], device=device)  
            
            if target_embedding.size(1) != output.size(1):
                print("Dimension mismatch.")
                continue

            loss = criterion(output, target_embedding, target_labels)
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
        print(batch_embeddings.size(0))
        if 200 + batch_index + i >= len(activation_cache_normal[layer_start]):
            print(len(activation_cache_normal[layer_start]))
            print("out of bounds")
            break
        primary_normal = torch.cat([activation_cache_normal[j][200 + batch_index + i][token_index].view(-1) for j in range(layer_start, layer_end + 1)]).unsqueeze(0).to(device)
        output = probe(primary_normal)
        target_embedding = batch_embeddings[i].unsqueeze(0)
        sim = F.cosine_similarity(output, target_embedding.unsqueeze(0))
        print(sim)
    batch_index += batch_embeddings.size(0)
