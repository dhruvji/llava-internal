import torch
import torch.nn as nn
import torch.optim as optim
import pickle

pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_100205.pkl'
pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_100205.pkl'

with open(pickle_file_path_normal, 'rb') as file:
    activation_cache_normal = pickle.load(file)

with open(pickle_file_path_red, 'rb') as file:
    activation_cache_red = pickle.load(file)

layer_index = 30

class BinaryProbe(nn.Module):
    def __init__(self, num_features):
        super(BinaryProbe, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)

num_features = activation_cache_normal[layer_index][0].numel()
probe = BinaryProbe(num_features)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(probe.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()  
    for instance in activation_cache_normal[layer_index]:
        instance_flat = instance.view(-1).unsqueeze(0)  
        label = torch.tensor([0.0])  
        output = probe(instance_flat).squeeze()  
        loss = criterion(output, label)
        loss.backward()  
        total_loss += loss.item()
    for instance in activation_cache_red[layer_index]:
        instance_flat = instance.view(-1).unsqueeze(0)
        label = torch.tensor([1.0])  
        output = probe(instance_flat).squeeze()
        loss = criterion(output, label)
        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    avg_loss = total_loss / (len(activation_cache_normal[layer_index]) + len(activation_cache_red[layer_index]))
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

print('Finished Training Probe')
