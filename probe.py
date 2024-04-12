import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random

import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_102149.pkl'
pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_102149.pkl'

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

num_features = activation_cache_normal[layer_index][0][activation_cache_normal[layer_index][0].size(0) - 3].numel() 
probe = BinaryProbe(num_features)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(probe.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()  
    for instance in activation_cache_normal[layer_index]:
        instance_flat = instance[activation_cache_normal[layer_index][0].size(0) - 3].view(-1).unsqueeze(0)  
        label = torch.tensor([[0.0]], device=instance_flat.device)  
        output = probe(instance_flat)  
        loss = criterion(output, label)  
        loss.backward()  
        total_loss += loss.item()

    for instance in activation_cache_red[layer_index]:
        instance_flat = instance[activation_cache_normal[layer_index][0].size(0) - 3].view(-1).unsqueeze(0)  
        label = torch.tensor([[1.0]], device=instance_flat.device) 
        output = probe(instance_flat)  
        loss = criterion(output, label)  
        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    avg_loss = total_loss / (len(activation_cache_normal[layer_index]) + len(activation_cache_red[layer_index]))
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')


print('Finished Training Probe')
os.makedirs(f"/data/dhruv_gautam/llava-internal/probes/", exist_ok=True)
model_save_path = f'/data/dhruv_gautam/llava-internal/probes/binary_probe{timestamp}.pth'
torch.save(probe.state_dict(), model_save_path)
print("Saved trained probe.")

probe_for_testing = BinaryProbe(num_features)
probe_for_testing.load_state_dict(torch.load(model_save_path))
probe_for_testing.eval()

# Load the test sets
pickle_file_path_normal_test = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_112213.pkl'
pickle_file_path_red_test = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_112213.pkl'

with open(pickle_file_path_normal_test, 'rb') as file:
    activation_cache_normal_test = pickle.load(file)
with open(pickle_file_path_red_test, 'rb') as file:
    activation_cache_red_test = pickle.load(file)

test_data = [(instance, 0) for instance in activation_cache_normal_test[layer_index]] + \
            [(instance, 1) for instance in activation_cache_red_test[layer_index]]
random.shuffle(test_data)  

predictions = []
true_labels = []
for instance, true_label in test_data:
    instance_flat = instance.view(-1).unsqueeze(0)
    output = probe_for_testing(instance_flat).squeeze()
    predicted_label = torch.sigmoid(output).round().item()
    predictions.append(predicted_label)
    true_labels.append(true_label)

accuracy = sum(pred == true for pred, true in zip(predictions, true_labels)) / len(true_labels)
print(f'Testing Accuracy: {accuracy}')

for pred, true in zip(predictions, true_labels):
    print(f'Predicted: {"Red" if pred == 1 else "Normal"}, Actual: {"Red" if true == 1 else "Normal"}')
