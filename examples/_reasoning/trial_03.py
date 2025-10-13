from unsloth import FastLanguageModel
from transformers import TextStreamer

max_seq_length = 2048 
lora_rank = 32 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./my_model",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

# Generate with the loaded model
_ = model.generate(
    **tokenizer("Fun fact:", return_tensors = "pt").to("cuda"),
    temperature = 0,
    max_new_tokens = 1024,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)

print("done")