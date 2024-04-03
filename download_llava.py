from transformers import AutoProcessor, AutoModelForPreTraining

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-1.5-7b-hf")

save_directory = "/data/dhruv_gautam/models/llava-v1.5-vicuna-7b"

processor.save_pretrained(save_directory)
model.save_pretrained(save_directory)
