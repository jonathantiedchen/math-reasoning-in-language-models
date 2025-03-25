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


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1', help='HuggingFace model path or name')
    parser.add_argument('--use_wandb', action='store_true', help='Use a model from Weights & Biases')
    parser.add_argument('--wandb_entity', type=str, help='W&B username or team name')
    parser.add_argument('--wandb_project', type=str, help='W&B project name')
    parser.add_argument('--wandb_model_artifact', type=str, help='W&B model artifact name (e.g., "run_name/model:v0")')
    parser.add_argument('--wandb_tokenizer_artifact', type=str, help='W&B tokenizer artifact name (if different from model)')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument("--test_run", action="store_true", help="Run with only the first X problems")
    parser.add_argument("--num_problems", type=int, default=10, help="Number of problems to use in test run")
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    
    if args.use_wandb:
        # Login to Weights & Biases (requires API key in environment variable or login)
        if not wandb.api.api_key:
            print("W&B API key not found. Please run 'wandb login' or set the WANDB_API_KEY environment variable.")
            return
            
        print(f"Downloading model from W&B: {args.wandb_model_artifact}")
        
        # Setup model download directory
        model_download_dir = os.path.join('wandb_models', args.wandb_model_artifact.replace('/', '_'))
        os.makedirs(model_download_dir, exist_ok=True)
        
        # Download model from W&B
        try:
            # Initialize W&B with entity and project
            api = wandb.Api()
            
            # Get model artifact
            model_artifact = api.artifact(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_model_artifact}")
            model_dir = model_artifact.download(root=model_download_dir)
            
            # Get tokenizer artifact (if specified, otherwise use the same as model)
            if args.wandb_tokenizer_artifact:
                tokenizer_artifact = api.artifact(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_tokenizer_artifact}")
                tokenizer_dir = tokenizer_artifact.download(root=model_download_dir)
            else:
                tokenizer_dir = model_dir
                
            # Load model and tokenizer from downloaded paths
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
        except Exception as e:
            print(f"Error downloading model from W&B: {e}")
            return
    else:
        # Use regular HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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

    results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        example = dataset[i]
        input_text = FEW_SHOT_PROMPT.format(question=example['question'])
        if i == 0:  # Print an example of the prompt for the first question
            print(f"EXAMPLE PROMPT: {input_text[:500]}...\n")
        
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, temperature=args.temp, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the final answer from the model's output
                output_text = output_text.split("A:")[-1].strip() 
                model_answer = extract_predicted_answer(output_text)
                model_answers.append({'text': output_text, 'numeric': model_answer})
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split("A:")[-1].strip() 
            model_answer = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': model_answer})

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        results.append({
            'question': example['question'],
            'gold_answer_text': example['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })
    
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")

    results.append({'accuracy': cnt / total})

    os.makedirs('eval_results/few_shot', exist_ok=True)
    
    # Determine model name for results file
    if args.use_wandb:
        model_name = args.wandb_model_artifact.split('/')[-1].replace(':', '_')
    else:
        model_name = args.model.split('/')[-1]
    
    result_file = f"eval_results/few_shot/{model_name}"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    if args.test_run:
        result_file += f"_test{args.num_problems}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
                

if __name__ == '__main__':
    main()