## vanilla
config_1 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": None,  # Reduced since we're using larger batches
        "gradient_checkpointing": False,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": False,
        "bf16": False,
        "optimizer": None,
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": False,
}

## 1. Modification - Gradient Accumulation
config_2 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": False,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": False,
        "bf16": False,
        "optimizer": None,
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": False,
}

## 2. Modification - Gradient Checkpointing
config_3 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": True,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": False,
        "bf16": False,
        "optimizer": None,
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": False,
}

### 3. Modification - Floating Datatypes - fp16
config_4 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": True,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": True,
        "bf16": False,
        "optimizer": None,
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": False,
}

### 4. Modification - Floating Datatypes - bf16
config_5 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": True,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": False,
        "bf16": True,
        "optimizer": None,
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": False,
}

### 5. Modification - Optimizer
config_6 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": True,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": True,
        "bf16": False,
        "optimizer": 'adafactor',
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": False,
}

### 6. Modification - Pin_Memory
config_7 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 10000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": True,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": True,
        "bf16": False,
        "optimizer": 'adafactor',
        "num_workers": None,  # Parallel data loading
        "dataloader_pin_memory": True,
}

### 7. Modification - Num_Workers
config_8 = {
        "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
        "dataset": "open-web-math",
        "streaming": True,
        "shuffle_buffer": 5000,  # Increased buffer size for better mixing
        "max_length": 1024,
        "max_steps": 5000,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "lr_scheduler_type": None, 

        ### GPU Training Optimization Parameter ###
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Reduced since we're using larger batches
        "gradient_checkpointing": True,
        "prefetch_factor": None,  # Prefetch factor for data loading
        "fp16": True,
        "bf16": False,
        "optimizer": 'adafactor',
        "num_workers": 4,  # Parallel data loading
        "dataloader_pin_memory": True,
}