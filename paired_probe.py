import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random

import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_112206.pkl' # 102149
pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_112206.pkl' #112206

with open(pickle_file_path_normal, 'rb') as file:
    activation_cache_normal = pickle.load(file)

with open(pickle_file_path_red, 'rb') as file:
    activation_cache_red = pickle.load(file)

layer_index = 30

class BinaryProbe(nn.Module):
    def __init__(self, num_features):
        super(BinaryProbe, self).__init__()
        self.linear1 = nn.Linear(num_features, 1)
        self.linear2 = nn.Linear(num_features, 1)

    def forward(self, x1, x2):
        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        return out1 - out2  

num_features = activation_cache_normal[layer_index][0].numel() 
probe = BinaryProbe(num_features)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(probe.parameters(), lr=0.001)

num_epochs = 50
batch_size = 1  
"""
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()
    min_length = min(len(activation_cache_normal[layer_index]), len(activation_cache_red[layer_index]))
    for i in range(min_length):
        normal_instance = activation_cache_normal[layer_index][i][activation_cache_normal[layer_index][i].size(0) - 3].view(-1).unsqueeze(0)
        red_instance = activation_cache_red[layer_index][i][activation_cache_red[layer_index][i].size(0) - 3].view(-1).unsqueeze(0)
        label = torch.tensor([0.0 if i % 2 == 0 else 1.0], device=normal_instance.device)
        output = probe(normal_instance, red_instance).squeeze(1)
        loss = criterion(output, label)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    avg_loss = total_loss / min_length
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

print('Finished Training Probe')
"""

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()
    min_length = min(len(activation_cache_normal[layer_index]), len(activation_cache_red[layer_index]))
    indices = list(range(min_length))
    random.shuffle(indices)  
    for i in indices:
        # Randomly choose order for normal and red instance
        if random.random() > 0.5:
            first_instance = activation_cache_normal[layer_index][i]
            second_instance = activation_cache_red[layer_index][i]
            label = torch.tensor([0.0], device=first_instance.device)  # Normal is 0, Red is 1
        else:
            first_instance = activation_cache_red[layer_index][i]
            second_instance = activation_cache_normal[layer_index][i] #[activation_cache_normal[layer_index][i].size(0) - 3]
            label = torch.tensor([1.0], device=first_instance.device)  # Red is 0, Normal is 1
        
        first_instance = first_instance.view(-1).unsqueeze(0)
        second_instance = second_instance.view(-1).unsqueeze(0)
        output = probe(first_instance, second_instance).squeeze(1)
        loss = criterion(output, label)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    avg_loss = total_loss / min_length
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

print('Finished Training Probe')

os.makedirs(f"/data/dhruv_gautam/llava-internal/probes/", exist_ok=True)
model_save_path = f'/data/dhruv_gautam/llava-internal/probes/binary_paired_probe{timestamp}.pth'
torch.save(probe.state_dict(), model_save_path)
print("Saved trained probe.")

probe_for_testing = BinaryProbe(num_features)  
probe_for_testing.load_state_dict(torch.load(model_save_path))
probe_for_testing.eval()  

pickle_file_path_normal_new_2 = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_112327.pkl'
pickle_file_path_red_new_2 = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_112327.pkl'

with open(pickle_file_path_normal_new_2, 'rb') as file:
    activation_cache_normal_new = pickle.load(file)

with open(pickle_file_path_red_new_2, 'rb') as file:
    activation_cache_red_new = pickle.load(file)
"""
predictions = []
true_labels = [0 if i % 2 == 0 else 1 for i in range(min(len(activation_cache_normal_new[layer_index]), len(activation_cache_red_new[layer_index])))]  
with torch.no_grad():
    for i in range(min(len(activation_cache_normal_new[layer_index]), len(activation_cache_red_new[layer_index]))):
        normal_instance = activation_cache_normal_new[layer_index][i].view(-1).unsqueeze(0)
        red_instance = activation_cache_red_new[layer_index][i].view(-1).unsqueeze(0)
        output = probe_for_testing(normal_instance, red_instance).squeeze()
        predicted_label = torch.sigmoid(output).round().item()
        predictions.append(predicted_label)

accuracy = sum(pred == true for pred, true in zip(predictions, true_labels)) / len(true_labels)
print(f'Testing Accuracy: {accuracy}')
"""
predictions = []
true_labels = []
with torch.no_grad():
    indices = list(range(min(len(activation_cache_normal_new[layer_index]), len(activation_cache_red_new[layer_index]))))
    random.shuffle(indices)  
    for i in indices:
        if random.random() > 0.5:
            first_instance = activation_cache_normal_new[layer_index][i]
            second_instance = activation_cache_red_new[layer_index][i]
            true_labels.append(0)  
        else:
            first_instance = activation_cache_red_new[layer_index][i]
            second_instance = activation_cache_normal_new[layer_index][i]
            true_labels.append(1)  
        
        first_instance = first_instance.view(-1).unsqueeze(0)
        second_instance = second_instance.view(-1).unsqueeze(0)
        output = probe_for_testing(first_instance, second_instance).squeeze()
        predicted_label = torch.sigmoid(output).round().item()
        predictions.append(predicted_label)

accuracy = sum(pred == true for pred, true in zip(predictions, true_labels)) / len(true_labels)
print(f'Testing Accuracy: {accuracy}')