#!pip install xformers
#!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#!pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


max_seq_length = 512  
dtype = (
    None 
)
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0, 
    bias="none", 
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False, 
    loftq_config=None
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["query"]
    solutions = examples["response"]
    texts = []
    for problem, solution in zip(instructions, solutions):
        text = alpaca_prompt.format(problem, solution) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset_path = "meta-math/MetaMathQA"
dataset = load_dataset(dataset_path, split="train")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False, 
    args=TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        warmup_ratio=0.03,
        max_steps=-1,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./results",
    )
)

trainer_stats = trainer.train()


