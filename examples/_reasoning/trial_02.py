from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 
lora_rank = 32 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

print("Model loaded successfully!")

# Enable inference mode
FastLanguageModel.for_inference(model)

# Get the device the model is on
device = next(model.parameters()).device
print(f"Model is on device: {device}")

# Define a chat template (adjust based on your model's expected format)
def format_chat_prompt(user_message, system_message=None):
    if system_message:
        return f"<|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
    else:
        return f"<|user|>\n{user_message}<|end|>\n<|assistant|>\n"

# Example 1: Simple text generation
def generate_response(prompt, max_new_tokens=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (response)
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return response.strip()

# Example 2: Chat-style inference
def chat_inference(user_message, system_message=None):
    prompt = format_chat_prompt(user_message, system_message)
    return generate_response(prompt)

# Example 3: Batch inference
def batch_inference(prompts, max_new_tokens=256):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    responses = []
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output[len(inputs['input_ids'][i]):], skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses

# Test the inference
if __name__ == "__main__":
    print("\n=== Testing Inference ===")
    
    try:
        # Test 1: Simple question
        user_question = "What is the capital of France?"
        response = chat_inference(user_question)
        print(f"Q: {user_question}")
        print(f"A: {response}")
        print("-" * 50)
        
        # Test 2: With system message
        system_msg = "You are a helpful assistant that provides concise answers."
        user_question = "Explain quantum computing in simple terms."
        response = chat_inference(user_question, system_msg)
        print(f"System: {system_msg}")
        print(f"Q: {user_question}")
        print(f"A: {response}")
        print("-" * 50)
        
        # Test 3: Batch inference
        questions = [
            "What is 2+2?",
            "Name a programming language.",
            "What color is the sky?"
        ]
        responses = batch_inference(questions, max_new_tokens=50)
        print("Batch inference results:")
        for q, a in zip(questions, responses):
            print(f"Q: {q}")
            print(f"A: {a}")
            print()
        
        print("Inference testing complete!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("This might be due to the chat template format not matching the model's expected format.")
        
        # Fallback: Simple text completion without chat formatting
        print("\nTrying simple text completion...")
        try:
            simple_prompt = "The capital of France is"
            response = generate_response(simple_prompt, max_new_tokens=50, temperature=0.1)
            print(f"Prompt: {simple_prompt}")
            print(f"Response: {response}")
        except Exception as e2:
            print(f"Simple completion also failed: {e2}")