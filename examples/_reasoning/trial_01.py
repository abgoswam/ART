
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower



def main():
    print("hello world")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Base",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.9, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )

    reasoning_start = "<start_working_out>" # Acts as <think>
    reasoning_end   = "<end_working_out>"   # Acts as </think>
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = \
        f"""You are given a problem.
        Think about the problem and provide your working out.
        Place it between {reasoning_start} and {reasoning_end}.
        Then, provide your solution between {solution_start}{solution_end}"""
    print(system_prompt)

    chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

    # Replace with out specific template:
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template

    # sample application of prompt template.
    print(tokenizer.apply_chat_template([
            {"role" : "user", "content" : "What is 1+1?"},
            {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
            {"role" : "user", "content" : "What is 2+2?"},
            ], 
        tokenize = False, 
        add_generation_prompt = True))

    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    dataset = dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # Try converting to number - if not, replace with NaN
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
    # Select only numbers
    dataset = dataset.iloc[np.where(is_number)[0]]

    print(dataset)

    def format_dataset(x):
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # Remove generated <think> and </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # Strip newlines on left and right
        thoughts = thoughts.strip()
        # Add our custom formatting
        final_prompt = \
            reasoning_start + thoughts + reasoning_end + \
            solution_start + expected_answer + solution_end
        return [
            {"role" : "system",    "content" : system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]

    dataset["Messages"] = dataset.apply(format_dataset, axis = 1)

    print(tokenizer.apply_chat_template(dataset["Messages"][0], tokenize = False))

    dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))

    dataset = dataset.loc[dataset["N"] <= max_seq_length/2].copy()

    print(dataset.shape)

    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
    dataset = Dataset.from_pandas(dataset)
    
    print(dataset)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 2, # Set this for 1 full training run.
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer.train()

    print("Training Complete.")

    text = tokenizer.apply_chat_template(
        dataset[0]["Messages"][:2],
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )

    from transformers import TextStreamer

    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        temperature = 0,
        max_new_tokens = 1024,
        streamer = TextStreamer(tokenizer, skip_prompt = False),
    )

    print("=" *30)

    # Save the model locally
    model.save_pretrained("my_model")
    tokenizer.save_pretrained("my_model")
    print("Model saved to my_model folder")

    # Load the saved model
    local_model, local_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "my_model",  # Path to your saved model
        max_seq_length = max_seq_length,
        load_in_4bit = False,
        fast_inference = True,
    )

    # Generate with the loaded model
    _ = local_model.generate(
        **local_tokenizer(text, return_tensors = "pt").to("cuda"),
        temperature = 0,
        max_new_tokens = 1024,
        streamer = TextStreamer(local_tokenizer, skip_prompt = False),
    )

if __name__ == "__main__":
    main()
    print("done with main")
