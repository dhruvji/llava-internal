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
from utils import batched_cache_activations_multimodal, hidden_cache_activations_multimodal

device = torch.device("cuda")
save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

class MultiLayerEmbeddingProbe(nn.Module):
    def __init__(self, input_features, output_size, hidden_size=1024):
        super(MultiLayerEmbeddingProbe, self).__init__()
        self.linear1 = nn.Linear(input_features, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)  
        self.activation1 = nn.GELU()  
        self.dropout1 = nn.Dropout(0.1)  
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)  
        self.activation2 = nn.GELU()  
        self.dropout2 = nn.Dropout(0.1)  
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)  
        self.activation3 = nn.GELU()  
        self.dropout3 = nn.Dropout(0.1)  
        self.linear4 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.activation1(self.norm1(self.linear1(x)))
        x = self.dropout1(x)
        x = self.activation2(self.norm2(self.linear2(x)))
        x = self.dropout2(x)
        x = self.activation3(self.norm3(self.linear3(x)))
        x = self.dropout3(x)
        x = self.linear4(x)
        return x

prompts = [
            "USER: <image>\nCan you describe the image?\nASSISTANT:",
            "USER: <image>\nPlease visualize the image colorized and describe it\nASSISTANT:",
            "USER: <image>\n\nASSISTANT:",
    ]

def clip_load():
    color_clip_embeddings_path = f'/data/dhruv_gautam/llava-internal/clip/original_embeddings010622.pth'
    clip_embeddings = torch.load(color_clip_embeddings_path)
    clip_embeddings = [batch.to(device) for batch in clip_embeddings]
    clip_embeddings_train = clip_embeddings[:190] #batched for 40 this means 200
    clip_embeddings_test = clip_embeddings[190:] # old 90
    return clip_embeddings_train, clip_embeddings_test

def image_load():
    image_dir = "/data/dhruv_gautam/coco/val2017/image_grey/original"
    image_dir_grey = "/data/dhruv_gautam/coco/val2017/image_grey/greyscale"
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
    image_paths_grey = [os.path.join(image_dir_grey, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

    image = []
    image_grey = []

    for img in image_paths[:1000]: 
        image.append(Image.open(img).convert("RGB"))
    for img in image_paths_grey[:1000]:
        image_grey.append(Image.open(img).convert("RGB"))
    return image, image_grey

def load_model(dir):
    processor = AutoProcessor.from_pretrained(dir)
    model = LlavaForConditionalGeneration.from_pretrained(dir)
    processor.tokenizer.padding_side = "left"
    return model, processor

def main():
    model, processor = load_model(save_directory)
    module_list = [f"model.language_model.model.layers[{i}]" for i in range(len(model.language_model.model.layers))]

    image, image_grey = image_load()

    print("Processing prompts and image...")
    inputs = processor([prompts[0] for _ in image], images=image, padding=True, return_tensors="pt")

    clip_embeddings_train, clip_embeddings_test = clip_load()

    # PARAM INIT
    embedding_size = 768
    num_epochs = 30
    layer_start = 10
    layer_end = 11

    # Calculate input_features by iterating through all the second dimensions
    inputs_batch = {key: val[:10].to(device) for key, val in inputs.items()}
    activation_cache_normal = hidden_cache_activations_multimodal(
        model=model,
        processor=processor,
        module_list_or_str=module_list,
        cache_input_output='output',
        inputs=inputs_batch,
        batch_size=10,
        token_idx=None,
    )

    input_features = sum(
        activation_cache_normal["hidden_states"][i][j].numel()
        for i in range(layer_start, layer_end + 1)
        for j in range(len(activation_cache_normal["hidden_states"][0]))
    )

    probe = MultiLayerEmbeddingProbe(input_features, embedding_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # TRAIN
    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        optimizer.zero_grad()
        batch_index = 0
        accumulation_steps = 4

        while batch_index < len(clip_embeddings_train):
            batch_embeddings = clip_embeddings_train[batch_index:batch_index + 10]
            inputs_batch = {key: val[batch_index:batch_index + 10].to(device) for key, val in inputs.items()}

            print("Caching Activations normal...")
            activation_cache_normal = hidden_cache_activations_multimodal(
                model=model,
                processor=processor,
                module_list_or_str=module_list,
                cache_input_output='output',
                inputs=inputs_batch,
                batch_size=10,
                token_idx=None,
            )

            for i in range(len(batch_embeddings)):
                primary_normal = torch.cat([
                    activation_cache_normal["hidden_states"][j][i].view(-1)
                    for j in range(layer_start, layer_end + 1)
                ]).unsqueeze(0).to(device)
                output = probe(primary_normal)
                target_embedding = batch_embeddings[i].unsqueeze(0)

                if target_embedding.size(1) != output.size(1):
                    print("Dimension mismatch.")
                    continue

                loss = criterion(output, target_embedding)
                loss = loss / accumulation_steps
                loss.backward()
                total_loss += loss.item() * accumulation_steps
                total_samples += 1

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Free up memory
            del activation_cache_normal
            torch.cuda.empty_cache()

            batch_index += 10

        avg_loss = total_loss / total_samples
        scheduler.step(avg_loss)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')
    print('Finished Training Probe')

    os.makedirs('/data/dhruv_gautam/llava-internal/probes/', exist_ok=True)
    model_save_path = f'/data/dhruv_gautam/llava-internal/probes/ist_clip_{timestamp}.pth'
    torch.save(probe.state_dict(), model_save_path)
    print("Saved trained probe.")

    # EVAL
    probe = MultiLayerEmbeddingProbe(input_features, embedding_size).to(device)
    probe.load_state_dict(torch.load(model_save_path))
    probe.eval()

    batch_index = 0
    with torch.no_grad():
        while batch_index < len(clip_embeddings_test):
            batch_embeddings = clip_embeddings_test[batch_index:batch_index + 10]
            inputs_batch = {key: val[batch_index:batch_index + 10].to(device) for key, val in inputs.items()}

            print("Caching Activations normal for evaluation...")
            activation_cache_normal = hidden_cache_activations_multimodal(
                model=model,
                processor=processor,
                module_list_or_str=module_list,
                cache_input_output='output',
                inputs=inputs_batch,
                batch_size=10,
                token_idx=None,
            )

            for i in range(len(batch_embeddings)):
                primary_normal = torch.cat([
                    activation_cache_normal["hidden_states"][j][i].view(-1)
                    for j in range(layer_start, layer_end + 1)
                ]).unsqueeze(0).to(device)
                output = probe(primary_normal)
                target_embedding = batch_embeddings[i].unsqueeze(0)
                sim = F.cosine_similarity(output, target_embedding.unsqueeze(0)).mean().item()
                print(sim)

            # Free up memory
            del activation_cache_normal
            torch.cuda.empty_cache()

            batch_index += 10

if __name__ == "__main__":
    main()