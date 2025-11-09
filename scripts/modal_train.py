
"""
Modal training script for finetuning LLMs with Unsloth.
Based on: https://modal.com/docs/examples/unsloth_finetune
"""

import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

# Create Modal App
app = modal.App("replai-unsloth-finetune")

# Container Image Configuration
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_HOME": "/model_cache"})
    # Mount dataset directory to upload local dataset files
    .add_local_dir("data/training-data", "/data/training-data")
)

with train_image.imports():
    # unsloth must be first!
    import unsloth  # noqa: F401,I001
    import datasets
    import torch
    import wandb
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import standardize_sharegpt

# Volume Configuration
model_cache_volume = modal.Volume.from_name(
    "replai-model-cache", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "replai-dataset-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "replai-checkpoints", create_if_missing=True
)

# GPU Configuration
GPU_TYPE = "L40S"  # Good balance of VRAM, CUDA cores, and clock speed
TIMEOUT_HOURS = 6
MAX_RETRIES = 3

# LoRA target modules for Qwen3/Hermes models
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Model and dataset configuration
    model_name: str
    dataset_name: str  # HuggingFace dataset name or local path
    max_seq_length: int
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False

    # Training hyperparameters
    optim: str = "adamw_8bit"
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    packing: bool = False
    use_gradient_checkpointing: str = "unsloth"
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_steps: int = 1000
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    num_train_epochs: Optional[float] = None
    max_grad_norm: float = 1.0

    # Optional configuration
    seed: int = 105
    experiment_name: Optional[str] = None
    enable_wandb: bool = True
    skip_eval: bool = False
    template: str = "qwen3"  # Chat template to use


# Data Processing Configuration
CONVERSATION_COLUMN = "messages"  # ShareGPT format column name
TEXT_COLUMN = "text"  # Output column for formatted text
TRAIN_SPLIT_RATIO = 0.9  # 90% train, 10% eval split
PREPROCESSING_WORKERS = 2


def format_chat_template(examples, tokenizer):
    """Format conversations using the tokenizer's chat template."""
    texts = []
    for conversation in examples[CONVERSATION_COLUMN]:
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {TEXT_COLUMN: texts}


def load_or_cache_dataset(config: TrainingConfig, paths: dict, tokenizer):
    """Load dataset from cache or download and process it."""
    dataset_cache_path = paths["dataset_cache"]

    if dataset_cache_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        train_dataset = datasets.load_from_disk(dataset_cache_path / "train")
        eval_dataset = datasets.load_from_disk(dataset_cache_path / "eval")
    else:
        print(f"Loading dataset: {config.dataset_name}")

        # Check if it's a local file path (starts with / or contains .json)
        import os
        is_local_file = (
            config.dataset_name.startswith("/") 
            or config.dataset_name.endswith(".json")
            or config.dataset_name.endswith(".jsonl")
        )
        
        if is_local_file:
            # Load from local JSON file
            # Normalize path: if relative, assume it's in /data/training-data
            if config.dataset_name.startswith("/"):
                dataset_path = config.dataset_name
            else:
                # Relative path - convert to absolute in container
                if "/" in config.dataset_name:
                    # Path like "data/training-data/replai.json" -> "/data/training-data/replai.json"
                    dataset_path = "/" + config.dataset_name
                else:
                    # Just filename like "replai.json" -> "/data/training-data/replai.json"
                    dataset_path = f"/data/training-data/{config.dataset_name}"
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Dataset file not found: {dataset_path}. "
                    f"Make sure to mount it in the Modal function. "
                    f"Expected path in container: /data/training-data/replai.json"
                )
            print(f"Loading from local file: {dataset_path}")
            dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")
        else:
            # Load from HuggingFace
            dataset = datasets.load_dataset(config.dataset_name, split="train")

        # Standardize to ShareGPT format if needed
        # Check for both "conversations" and "messages" columns
        has_conversations = "conversations" in dataset.column_names
        has_messages = "messages" in dataset.column_names
        
        if has_conversations:
            print("Found 'conversations' column, standardizing ShareGPT format...")
            dataset = standardize_sharegpt(dataset)
            # After standardization, it should have "messages" column
            if "messages" not in dataset.column_names:
                print("Warning: Standardization didn't create 'messages' column")
        elif has_messages:
            print("Found 'messages' column, using as-is")
        else:
            # If not in ShareGPT format, assume it's already formatted
            # You may need to adjust this based on your dataset format
            print(f"Dataset columns: {dataset.column_names}")
            print("Dataset not in ShareGPT format, using as-is")

        # Split into training and evaluation sets
        dataset = dataset.train_test_split(
            test_size=1.0 - TRAIN_SPLIT_RATIO, seed=config.seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Apply chat template formatting
        # Check for both "conversations" and "messages" columns
        has_conversations_col = "conversations" in train_dataset.column_names
        has_messages_col = "messages" in train_dataset.column_names
        
        if has_conversations_col or has_messages_col:
            # Use the column that exists
            conv_col = "messages" if has_messages_col else "conversations"
            print(f"Formatting datasets with chat template using '{conv_col}' column...")
            
            # Create a wrapper function that uses the correct column
            # Using non-batched processing for reliability
            def format_single_conversation(example, tokenizer, column_name):
                conversation = example[column_name]
                try:
                    formatted_text = tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=False
                    )
                    if formatted_text is None:
                        formatted_text = ""
                except Exception as e:
                    print(f"Error formatting conversation: {e}")
                    formatted_text = ""
                
                return {TEXT_COLUMN: formatted_text}
            
            print("Processing train dataset (this may take a while)...")
            train_dataset = train_dataset.map(
                format_single_conversation,
                fn_kwargs={"tokenizer": tokenizer, "column_name": conv_col},
                remove_columns=[
                    col
                    for col in train_dataset.column_names
                    if col != TEXT_COLUMN
                ],
                desc="Formatting training data",
            )
            print("Processing eval dataset...")
            eval_dataset = eval_dataset.map(
                format_single_conversation,
                fn_kwargs={"tokenizer": tokenizer, "column_name": conv_col},
                remove_columns=[
                    col for col in eval_dataset.column_names if col != TEXT_COLUMN
                ],
                desc="Formatting eval data",
            )
        elif CONVERSATION_COLUMN in train_dataset.column_names:
            print("Formatting datasets with chat template...")
            train_dataset = train_dataset.map(
                format_chat_template,
                batched=True,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=PREPROCESSING_WORKERS,
                remove_columns=[
                    col
                    for col in train_dataset.column_names
                    if col != TEXT_COLUMN
                ],
            )
            eval_dataset = eval_dataset.map(
                format_chat_template,
                batched=True,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=PREPROCESSING_WORKERS,
                remove_columns=[
                    col for col in eval_dataset.column_names if col != TEXT_COLUMN
                ],
            )

        # Cache the processed datasets
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(dataset_cache_path / "train")
        eval_dataset.save_to_disk(dataset_cache_path / "eval")
        dataset_cache_volume.commit()

    return train_dataset, eval_dataset


def get_structured_paths(config: TrainingConfig):
    """Create structured paths within the mounted volumes."""
    dataset_cache_path = (
        pathlib.Path("/dataset_cache")
        / "datasets"
        / config.dataset_name.replace("/", "--")
    )
    checkpoint_path = (
        pathlib.Path("/checkpoints") / "experiments" / config.experiment_name
    )

    return {
        "dataset_cache": dataset_cache_path,
        "checkpoints": checkpoint_path,
    }


def setup_model_for_training(model, config: TrainingConfig):
    """Configure the model with LoRA adapters for efficient finetuning."""
    print("Configuring LoRA for training...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )
    return model


def create_training_arguments(config: TrainingConfig, output_dir: str):
    """Create training arguments for the SFTTrainer."""
    return TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps if config.num_train_epochs is None else None,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="no" if config.skip_eval else "steps",
        save_strategy="steps",
        do_eval=not config.skip_eval,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        output_dir=output_dir,
        report_to="wandb" if config.enable_wandb else None,
        seed=config.seed,
    )


def check_for_existing_checkpoint(paths: dict):
    """Check if there's an existing checkpoint to resume training from."""
    checkpoint_dir = paths["checkpoints"]
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        print(f"Found existing checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    return None


@app.function(
    image=train_image,
    gpu=modal.gpu.H100(count=8),
    volumes={
        "/model_cache": model_cache_volume,
        "/dataset_cache": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=TIMEOUT_HOURS * 3600,
    retries=MAX_RETRIES,
    # Optional wandb secret - create with: modal secret create wandb-secret WANDB_API_KEY=your_key_here
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def finetune(config: TrainingConfig):
    """Main training function."""
    # Generate experiment name if not provided
    if config.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config.experiment_name = f"{config.model_name.split('/')[-1]}-{timestamp}"

    print(f"Starting finetuning experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")

    # Initialize wandb if enabled
    if config.enable_wandb:
        import os
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(
                project="replai-finetune",
                name=config.experiment_name,
                config=config.__dict__,
            )
        else:
            print("Warning: WANDB_API_KEY not found. Disabling wandb logging.")
            print("To enable wandb, set WANDB_API_KEY as a Modal secret or environment variable.")
            config.enable_wandb = False

    # Get paths
    paths = get_structured_paths(config)

    # Load model
    print(f"Loading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        trust_remote_code=True,
    )

    # Set chat template
    FastLanguageModel.for_training(model)

    # Load and process dataset
    train_dataset, eval_dataset = load_or_cache_dataset(config, paths, tokenizer)

    # Setup LoRA
    model = setup_model_for_training(model, config)

    # Check for existing checkpoint
    resume_from_checkpoint = check_for_existing_checkpoint(paths)

    # Create training arguments
    output_dir = str(paths["checkpoints"])
    training_args = create_training_arguments(config, output_dir)

    # Create trainer
    # Verify dataset has the required "text" column after processing
    print(f"Dataset columns after processing: {train_dataset.column_names}")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of eval examples: {len(eval_dataset)}")
    
    # Check if text column exists
    if TEXT_COLUMN not in train_dataset.column_names:
        raise ValueError(
            f"Dataset does not have '{TEXT_COLUMN}' column after processing. "
            f"Available columns: {train_dataset.column_names}. "
            f"This suggests the chat template formatting failed."
        )
    
    # Verify text column has valid data
    print(f"Checking first few examples in {TEXT_COLUMN} column...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        text_value = sample[TEXT_COLUMN]
        if not isinstance(text_value, str):
            raise ValueError(f"Example {i}: {TEXT_COLUMN} column contains non-string: {type(text_value)}")
        if not text_value or len(text_value) == 0:
            raise ValueError(f"Example {i}: {TEXT_COLUMN} column is empty")
        print(f"  Example {i}: {len(text_value)} characters, starts with: {text_value[:100]}...")
    
    # Use the text column for training
    dataset_text_field = TEXT_COLUMN
    formatting_func = None
    print(f"✓ Using dataset_text_field='{dataset_text_field}' for training")
    
    # Create trainer with explicit parameters
    print(f"Creating SFTTrainer with:")
    print(f"  - dataset_text_field: {dataset_text_field}")
    print(f"  - formatting_func: {formatting_func}")
    print(f"  - packing: {config.packing}")
    print(f"  - max_seq_length: {config.max_seq_length}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not config.skip_eval else None,
        args=training_args,
        dataset_text_field=dataset_text_field,  # Should be "text"
        formatting_func=None,  # Don't use formatting_func since we have text column
        max_seq_length=config.max_seq_length,
        packing=config.packing,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("Saving final model...")
    final_model_path = paths["checkpoints"] / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Commit volumes
    checkpoint_volume.commit()
    model_cache_volume.commit()

    print(f"Training completed! Model saved to {final_model_path}")

    if config.enable_wandb:
        wandb.finish()

    return config.experiment_name


@app.local_entrypoint()
def main(
    # Model and dataset configuration
    model_name: str = "NousResearch/Hermes-4-14B",
    dataset_name: str = "/data/training-data/replai.json",  # Local file path or HuggingFace dataset name
    max_seq_length: int = 131072,  # Match your train.sh cutoff_len
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    # LoRA hyperparameters
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_bias: str = "none",
    use_rslora: bool = False,
    # Training hyperparameters
    optim: str = "adamw_8bit",
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    packing: bool = False,
    use_gradient_checkpointing: str = "unsloth",
    learning_rate: float = 1e-5,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    max_steps: Optional[int] = None,
    num_train_epochs: Optional[float] = 3.0,
    save_steps: int = 200,
    eval_steps: int = 200,
    logging_steps: int = 10,
    max_grad_norm: float = 1.0,
    # Optional configuration
    seed: int = 105,
    experiment_name: Optional[str] = None,
    disable_wandb: bool = False,
    skip_eval: bool = False,
    template: str = "qwen3",
):
    """Entry point for training."""
    config = TrainingConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_bias=lora_bias,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        optim=optim,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        packing=packing,
        use_gradient_checkpointing=use_gradient_checkpointing,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        experiment_name=experiment_name,
        enable_wandb=not disable_wandb,
        skip_eval=skip_eval,
        template=template,
    )

    print(f"Starting finetuning experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"LoRA configuration: rank={config.lora_r}, alpha={config.lora_alpha}")
    print(
        f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}"
    )
    if config.max_steps:
        print(f"Training steps: {config.max_steps}")
    else:
        print(f"Training epochs: {config.num_train_epochs}")

    experiment_name = finetune.remote(config)
    print(f"Training completed successfully: {experiment_name}")
