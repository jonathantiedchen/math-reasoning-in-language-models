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


# Original 8-shot prompt
EIGHT_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
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

# New 4-shot prompt for GPT-2
FOUR_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""


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
    parser.add_argument("--test_run", action="store_true", help="Run with only the first X problems")
    parser.add_argument("--num_problems", type=int, default=10, help="Number of problems to use in test run")
    parser.add_argument("--use_cot_prompt", action="store_true", help="Enable Chain-of-Thought prompting")
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    
    model_artifact = args.model if not args.wandb_artifact else args.wandb_artifact

    # Start a WandB run for logging metrics
    run = wandb.init(
        project=f"gsm8k_evaluation_few_shot",
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
    
    if args.wandb_artifact:
        # Login to Weights & Biases (requires API key in environment variable or login)
        if not wandb.api.api_key:
            print("W&B API key not found. Please run 'wandb login' or set the WANDB_API_KEY environment variable.")
            return
            
        print(f"Downloading model from W&B: {args.wandb_artifact}")
        
        # Setup model download directory
        artifact_safe_name = args.wandb_artifact.replace('/', '_').replace(':', '_')
        model_download_dir = os.path.join('wandb_models', artifact_safe_name)
        os.makedirs(model_download_dir, exist_ok=True)
        
        # Download model from W&B using run.use_artifact approach
        try:
            artifact = run.use_artifact(args.wandb_artifact, type='model')
            model_dir = artifact.download(root=model_download_dir)
            
            # Get tokenizer artifact (if specified, otherwise use the same as model)
            if args.wandb_tokenizer_artifact:
                tokenizer_artifact = run.use_artifact(args.wandb_tokenizer_artifact, type='model')
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

    # Check if we're using a GPT-2 model to determine which prompt to use
    is_gpt2 = "gpt2" in model_artifact.lower()
    if is_gpt2:
        print("Using 4-shot prompt for GPT-2 model")
        prompt_template = FOUR_SHOT_PROMPT
    else:
        print("Using 8-shot prompt for non-GPT-2 model")
        prompt_template = EIGHT_SHOT_PROMPT

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]

    results = []

    client = weave.init("gsm8k_evaluation_few_shot")
        
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
                
        input_text = prompt_template.format(question=example['question'])
        if i == 0:  # Print an example of the prompt for the first question
            print(f"EXAMPLE PROMPT: {input_text[:500]}...\n")

        # Set appropriate max_length based on model
        max_length = 1024 if is_gpt2 else model.config.max_position_embeddings
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            temperature=args.temp, 
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
    
    os.makedirs('eval_results/few_shot', exist_ok=True)
    
    # Determine model name for results file
    if args.wandb_artifact:
        model_name = args.wandb_artifact.split('/')[-1].replace(':', '_')
    else:
        model_name = args.model.split('/')[-1]
    
    result_file = f"eval_results/few_shot/{model_name}"
    # Add shot count to the filename
    if is_gpt2:
        result_file += "_4shot"
    else:
        result_file += "_8shot"
        
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    if args.test_run:
        result_file += f"_test{args.num_problems}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
                
    # Finish the run
    wandb.finish()
    
if __name__ == '__main__':
    main()