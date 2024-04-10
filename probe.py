import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

pickle_file_path = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_100205.pkl'

with open(pickle_file_path, 'rb') as file:
    activation_cache = pickle.load(file)

pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_100205.pkl'

with open(pickle_file_path_red, 'rb') as file:
    activation_cache_red = pickle.load(file)

layer_index = x  
activations_normal = activation_cache[layer_index].view(activation_cache[layer_index].size(0), -1)
activations_red = activation_cache_red[layer_index].view(activation_cache_red[layer_index].size(0), -1)

labels = torch.cat((torch.zeros(activations_normal.size(0)), torch.ones(activations_red.size(0))), dim=0)

activations = torch.cat((activations_normal, activations_red), dim=0)

dataset = TensorDataset(activations, labels)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

class BinaryProbe(nn.Module):
    def __init__(self, num_features):
        super(BinaryProbe, self).__init__()
        self.linear = nn.Linear(num_features, 1)  

    def forward(self, x):
        return self.linear(x)

num_features = activations.shape[1] 
probe = BinaryProbe(num_features)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(probe.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()

        outputs = probe(inputs).squeeze(1)  
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print('Finished Training Probe')
