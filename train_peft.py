import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import os

def prepare_model():
    # Load base model 
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model
    model = prepare_model_for_kbit_training(model)
    
    # checkpoints
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Lora config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # PEFT model
    model = get_peft_model(model, lora_config)
    
    
    model.train()
    
    # trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def format_chat(example, tokenizer):
    # Format the conversation into chat format
    conversation = f"User: {example['input']}\nAssistant: {example['output']}"
    
    # Tokenize the text
    tokenized = tokenizer(
        conversation,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "labels": tokenized["input_ids"][0].clone()
    }

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = prepare_model()
    
    print("Loading MentalChat16K dataset...")

    # Load  MentalChat16K dataset
    dataset = load_dataset("ShenLab/MentalChat16K")
    
    # Split dataset into train and validation (90/10 split)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Format datasets
    train_dataset = train_dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="mental_health_peft_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_grad_norm=0.3,
        logging_steps=10,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        warmup_ratio=0.03,
        group_by_length=True,
        report_to="none",
        gradient_checkpointing=True,
        fp16=True,
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    
    print("Training complete!")

if __name__ == "__main__":
    main() 