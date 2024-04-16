import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

device = torch.device("cuda")

class HybridEmbeddingProbe(nn.Module):
    def __init__(self, primary_features, auxiliary_features, embedding_size):
        super(HybridEmbeddingProbe, self).__init__()
        self.primary_linear = nn.Linear(primary_features, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.auxiliary_linear = nn.Linear(auxiliary_features, 1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.combiner = nn.Linear(2048, embedding_size)
        self.final_relu = nn.ReLU()

    def forward(self, primary, auxiliary):
        primary_out = self.dropout1(self.relu1(self.primary_linear(primary)))
        auxiliary_out = self.dropout2(self.relu2(self.auxiliary_linear(auxiliary)))
        combined = torch.cat([primary_out, auxiliary_out], dim=1)
        final_output = self.final_relu(self.combiner(combined))
        return final_output

embedding_size = 4096
num_epochs = 300
token_index = -3 
context_indices = slice(-13, -8)  # context tokens indices (adjust as needed)

pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/visualize_main_cache150535.pkl'
pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/red/visualize_main_cache150535.pkl'

d_pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/describe_main_cache152326.pkl'
d_pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/red/describe_main_cache152326.pkl'

with open(pickle_file_path_normal, 'rb') as file:
    activation_cache_normal = pickle.load(file)

with open(d_pickle_file_path_red, 'rb') as file:
    activation_cache_red = pickle.load(file)

layer_index = 20
primary_features = activation_cache_normal[layer_index][0][token_index].numel()
auxiliary_features = activation_cache_normal[layer_index][0][context_indices].reshape(-1).numel()

probe = HybridEmbeddingProbe(primary_features, auxiliary_features, embedding_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(probe.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()
    for instance_normal, instance_red in zip(activation_cache_normal[layer_index], activation_cache_red[layer_index]):
        
        primary_normal = instance_normal[token_index].view(-1).unsqueeze(0)
        primary_red = instance_red[token_index].view(-1).unsqueeze(0)
        auxiliary_normal = instance_normal[context_indices].reshape(-1).unsqueeze(0)

        primary_normal = primary_normal.to(device)
        primary_red = primary_red.to(device)
        auxiliary_normal = auxiliary_normal.to(device)
        
        output = probe(primary_normal, auxiliary_normal)
        loss = criterion(output, primary_red)
        loss.backward()
        total_loss += loss.item()
        
    optimizer.step()
    avg_loss = total_loss / len(activation_cache_normal[layer_index])
    scheduler.step(avg_loss)  
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

print('Finished Training Probe')

os.makedirs(f"/data/dhruv_gautam/llava-internal/probes/", exist_ok=True)
model_save_path = f'/data/dhruv_gautam/llava-internal/probes/edit_probe{timestamp}.pth'
torch.save(probe.state_dict(), model_save_path)
print("Saved trained probe.")

probe_for_testing = HybridEmbeddingProbe(primary_features, auxiliary_features, embedding_size)
probe_for_testing.load_state_dict(torch.load(model_save_path))
probe_for_testing = probe_for_testing.to(device)
probe_for_testing.eval()

#pickle_file_path_normal_test = '/data/dhruv_gautam/llava-internal/caches/reg/visualize_test_cache150623.pkl'
#pickle_file_path_red_test = '/data/dhruv_gautam/llava-internal/caches/red/visualize_test_cache150623.pkl'
d_pickle_file_path_normal_test = '/data/dhruv_gautam/llava-internal/caches/reg/describe_test_cache152318.pkl'
d_pickle_file_path_red_test = '/data/dhruv_gautam/llava-internal/caches/red/describe_test_cache152318.pkl'

with open(d_pickle_file_path_normal_test, 'rb') as file:
    activation_cache_normal_test = pickle.load(file)
with open(d_pickle_file_path_red_test, 'rb') as file:
    activation_cache_red_test = pickle.load(file)

print("Number of normal instances:", len(activation_cache_normal_test[layer_index]))
print("Number of red instances:", len(activation_cache_red_test[layer_index]))

predefined_outputs = torch.stack([activation_cache_red_test[layer_index][i][token_index].view(-1) for i in range(len(activation_cache_red_test[layer_index]))]).to(device)
all_test_data = []
correct_labels = []  

for i in range(len(activation_cache_normal_test[layer_index])):
    # Prepare test data
    test_data = [
        (activation_cache_normal_test[layer_index][i][token_index], 
         activation_cache_normal_test[layer_index][i][context_indices].reshape(-1), 0, 'normal'),  
        (activation_cache_red_test[layer_index][i][token_index], 
         activation_cache_red_test[layer_index][i][context_indices].reshape(-1), 1, 'target_red')  
    ]
    remaining_indices = list(set(range(len(activation_cache_red_test[layer_index]))) - {i})
    random_indices = random.sample(remaining_indices, 3)
    test_data += [
        (activation_cache_red_test[layer_index][j][token_index], 
         activation_cache_red_test[layer_index][j][context_indices].reshape(-1), 2, 'random_red') for j in random_indices
    ]
    random.shuffle(test_data)
    correct_index = next(index for index, (_, _, label, identifier) in enumerate(test_data) if identifier == 'target_red')
    all_test_data.append(test_data)
    correct_labels.append(correct_index)

total_correct = 0  

for group_index, test_group in enumerate(all_test_data):
    correct_index = correct_labels[group_index]  
    predictions = []
    
    for index, (primary, auxiliary, label, identifier) in enumerate(test_group):
        primary_flat = primary.view(-1).unsqueeze(0).to(device)
        auxiliary_flat = auxiliary.unsqueeze(0).to(device)
        embedding = probe_for_testing(primary_flat, auxiliary_flat).squeeze(0)
        similarities = torch.cosine_similarity(embedding.unsqueeze(0), predefined_outputs)
        predicted_label = torch.argmax(similarities).item()
        predictions.append(predicted_label)

    predicted_correctly = (predictions[correct_index] == 1) 
    if predicted_correctly:
        total_correct += 1
    print(f'Group {group_index + 1} predicted correctly: {predicted_correctly}')

success_rate = total_correct / len(all_test_data)
print(f'Total Correctly Predicted Groups: {total_correct}/{len(all_test_data)}')
print(f'Success Rate: {success_rate:.2f}')