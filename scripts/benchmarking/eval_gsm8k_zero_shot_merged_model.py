# code used from repository: https://github.com/tianlwang/eval_gsm8k/tree/main?tab=readme-ov-file#
import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList
)
from utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from collections import Counter
import json
import wandb
import weave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1', help='HuggingFace model path or name')
    parser.add_argument('--wandb_artifact', type=str, help='Full W&B artifact link (e.g., "username/project/model:v0")')
    parser.add_argument('--wandb_tokenizer_artifact', type=str, help='W&B tokenizer artifact (if different from model)')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=float, default=512)
    parser.add_argument("--use_cot_prompt", action="store_true")
    parser.add_argument("--test_run", action="store_true", help="Run with only the first X problems")
    parser.add_argument("--num_problems", type=int, default=10, help="Number of problems to use in test run")
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    model_artifact = args.model if not args.wandb_artifact else args.wandb_artifact
    # Start a WandB run for logging metrics
    wandb_run = wandb.init(
        entity = "master_thesis_math_lm",
        project=f"gsm8k_evaluation_zero_shot",
        name= f"gsm8k_evaluation_{model_artifact}",
        config={
            "model": model_artifact,
            "use_cot_prompt": args.use_cot_prompt,
            "use_majority_vote": args.use_majority_vote,
            "n_votes": args.n_votes,
            "temperature": args.temp,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens
        }
    )
    
    
    ### LOADING ADAPTER FROM HUGGINGFACE AND MERGE WITH BASE MODEL
    # Load the base model
    print("Loading base GPT-2 model...")
    base_model_name = "gpt2-large"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # GPT-2 tokenizer doesn't have a padding token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id

    artifact = run.use_artifact(args.wandb_artifact, type='model')
    adapter_dir = artifact.download()
    print(f"Files in adapter directory: {os.listdir(adapter_dir)}")
    
    # If adapter path is provided, load and merge the adapter
    if args.wandb_artifact:
        print(f"Loading LoRA adapter from {args.wandb_artifact}...")
        try:
            # Load the adapter onto the base model
            adapted_model = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                is_trainable=False  # Set to False since we'll merge it
            )
            
            print("Successfully loaded LoRA adapter")
            
            # Merge the adapter with the base model
            print("Merging LoRA adapter into base model...")
            model = adapted_model.merge_and_unload()
            
            print("Successfully merged adapter with base model")
            
            # Free up memory from the original models
            del base_model
            del adapted_model
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error loading/merging adapter: {e}")
            print("Falling back to using base model only")
            model = base_model
    else:
        # If no adapter is provided, use the base model as is
        model = base_model
    
    # Set model to evaluation mode
    model.eval()
    model.to("cuda")
    
    # Add after model loading to confirm device placement
    print(f"Model device: {next(model.parameters()).device}")

    ### LOAD DATASET
    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')
    
    # Take only the first X problems if in test run mode
    if args.test_run:
        dataset = dataset.select(range(min(args.num_problems, len(dataset))))
        print(f'Running in test mode with the first {len(dataset)} problems')
    
    datasize = len(dataset)
    print('gsm8k dataset size:', datasize) 

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]

    ## EVALUATION LOOP
    results = []
    client = weave.init("master_thesis_math_lm/gsm8k_evaluation_zero_shot")
    for i in tqdm(range(datasize), desc='Evaluating'):
        example = dataset[i]
        call = client.create_call(
            op=f"prompt_{i}", 
            inputs={
                "model": model_artifact, 
                "question": example['question'], 
                "temperature":args.temp, 
                "top_k":args.top_k,
                "top_p":args.top_p,
                "max_new_tokens": args.max_new_tokens
            })
        if args.use_cot_prompt:
            input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
        else:
            input_text = 'Q: ' + example['question'] + '\nA:'
        print(f"MODEL INPUT: {input_text}")
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])
        
        # Define a stopping condition for generation
        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        temperature=args.temp, 
                        top_k=args.top_k,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens, 
                        do_sample=True, 
                        pad_token_id=tokenizer.eos_token_id, 
                        stopping_criteria=stopping_criteria_list
                    )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the final answer from the model's output
                output_text = output_text.split("A:")[-1].strip() 
                model_answer = extract_predicted_answer(output_text)
                model_answers.append({'text': output_text, 'numeric': model_answer})
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens, 
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria_list
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split("A:")[-1].strip() 
            model_answer = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': model_answer})

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        outputs = {
            'question': example['question'],
            'gold_answer_text': example['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        }
        results.append(outputs)
        client.finish_call(call, output=outputs)
    
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    accuracy = cnt/total
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    
    results.append({'accuracy': cnt / total})
    
    # Log the accuracy to WandB
    wandb.log({"accuracy": accuracy, "correct_count": cnt, "total_samples": total})

    os.makedirs('eval_results/zero_shot', exist_ok=True)
    
    # Determine model name for results file
    if args.wandb_artifact:
        model_name = args.wandb_artifact.split('/')[-1].replace(':', '_')
    else:
        model_name = args.model.split('/')[-1]
    
    result_file = f"eval_results/zero_shot/{model_name}"
    if args.use_cot_prompt:
        result_file += "_cot"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    if args.test_run:
        result_file += f"_test{args.num_problems}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")

    #Finish Run:
    wandb.finish()

if __name__ == '__main__':
    main()

