import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
import datetime
timestamp = datetime.datetime.now().strftime("%d%H%M")

class EmbeddingProbe(nn.Module):
    def __init__(self, num_features, embedding_size):
        super(EmbeddingProbe, self).__init__()
        self.linear = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        return self.linear(x)

embedding_size = 10  
num_epochs = 350
batch_size = 1

pickle_file_path_normal = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_112206.pkl'
pickle_file_path_red = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_112206.pkl'

with open(pickle_file_path_normal, 'rb') as file:
    activation_cache_normal = pickle.load(file)

with open(pickle_file_path_red, 'rb') as file:
    activation_cache_red = pickle.load(file)

layer_index = 30
num_features = activation_cache_normal[layer_index][0].numel()

probe = EmbeddingProbe(num_features, embedding_size)
criterion = nn.MSELoss()                                                                                                    
optimizer = optim.Adam(probe.parameters(), lr=0.001)

predefined_outputs = torch.randn(5, embedding_size) 

for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()  
    for instance in activation_cache_normal[layer_index]:
        instance_flat = instance.view(-1).unsqueeze(0)
        output = probe(instance_flat)
        target = torch.randn(1, embedding_size)  
        loss = criterion(output, target)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    avg_loss = total_loss / len(activation_cache_normal[layer_index])
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

print('Finished Training Probe')

os.makedirs(f"/data/dhruv_gautam/llava-internal/probes/", exist_ok=True)
model_save_path = f'/data/dhruv_gautam/llava-internal/probes/edit_probe{timestamp}.pth'
torch.save(probe.state_dict(), model_save_path)
print("Saved trained probe.")

probe_for_testing = EmbeddingProbe(num_features, embedding_size)
probe_for_testing.load_state_dict(torch.load(model_save_path))
probe_for_testing.eval()

pickle_file_path_normal_test = '/data/dhruv_gautam/llava-internal/caches/reg/activation_cache2_112327.pkl'
pickle_file_path_red_test = '/data/dhruv_gautam/llava-internal/caches/red/activation_cache2_112327.pkl'

with open(pickle_file_path_normal_test, 'rb') as file:
    activation_cache_normal_test = pickle.load(file)
with open(pickle_file_path_red_test, 'rb') as file:
    activation_cache_red_test = pickle.load(file)

all_test_data = []
for i in range(len(activation_cache_normal_test[layer_index])):
    test_data = [
        (activation_cache_normal_test[layer_index][i], 0),  # normal image
        (activation_cache_red_test[layer_index][i], 1)      # target red image
    ]
    remaining_indices = list(set(range(len(activation_cache_red_test[layer_index]))) - {i})
    random_indices = random.sample(remaining_indices, 3)
    test_data += [(activation_cache_red_test[layer_index][j], 2) for j in random_indices]   # 3 random reds
    random.shuffle(test_data)

    all_test_data.append(test_data)

for test_group in all_test_data:
    predictions = []
    true_labels = []
    for instance, true_label in test_group:
        instance_flat = instance.view(-1).unsqueeze(0)
        embedding = probe_for_testing(instance_flat).squeeze(0)
        similarities = torch.cosine_similarity(embedding.unsqueeze(0), predefined_outputs)
        predicted_label = torch.argmax(similarities).item()
        predictions.append(predicted_label)
        true_labels.append(true_label)

    accuracy = sum(pred == true for pred, true in zip(predictions, true_labels)) / len(true_labels)
    print(f'Testing Accuracy for this group based on highest cosine similarity: {accuracy}')