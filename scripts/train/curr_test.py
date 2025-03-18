import os
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

# Initialize Weights & Biases
run = wandb.init(project="gpt2-math", name="curriculum-learning-sft")
artifact = run.use_artifact('master_thesis_math_lm/gpt2-math/gpt2-math-model:v0', type='model')

# Download the artifact and get the directory path
artifact_dir = artifact.download()

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(artifact_dir)
model = GPT2LMHeadModel.from_pretrained(artifact_dir)

# Load model and tokenizer
model_name = "gpt2"


tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Data loading functions (assuming they return lists of dicts with 'text' field)
data_loaders = {
    "1_ASDiv": load_asdiv_data,
    "2_ParaMAWPS": load_paramawps_data,
    "3_SVAMP": load_svamp_data,
    "4_Dmath": load_dmath_data,
    "5_AQuA": load_aqua_data
}

data_root = "your_data_root_path_here"
data_paths = [
    os.path.join(data_root, "data", "curriculum_learning", folder, file)
    for folder, file in zip(
        ["1_ASDiv", "2_ParaMAWPS", "3_SVAMP", "4_Dmath", "5_AQuA"],
        ["ASDiv.xml", "ParaMAWPS_trainset.json", "SVAMP.json", "dmath_train.json", "AQuA_train.json"]
    )
]

def prepare_dataset(data_loader, data_path):
    data = data_loader(data_path)
    return Dataset.from_list([{"text": entry["text"]} for entry in data])

training_args = TrainingArguments(
    output_dir="./models/gpt2-math-curriculum",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=10,
    bf16=torch.cuda.is_available(),
    report_to="wandb",
)

for path, (name, loader) in zip(data_paths, data_loaders.items()):
    print(f"Training on {name}...")
    dataset = prepare_dataset(loader, path)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"./models/gpt2-math-curriculum/{name}")

wandb.finish()
