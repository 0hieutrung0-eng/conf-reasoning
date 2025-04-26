#!/usr/bin/env python3
import os
import math
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm, trange
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = "/tmp"

# Number of rollouts per question
NUM_ROLLOUTS = 16
# Save checkpoint every N entries
SAVE_INTERVAL = 100

def load_mmlu_dataset():
    """Load MMLU dataset from HuggingFace"""
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all")
    return dataset["auxiliary_train"]

def initialize_model():
    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=64,
    )
    
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
    return tokenizer, llm, sampling_params

def format_question(question, choices):
    """Format the question for the model"""
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""Question: {question}

Choices:
{choices_text}

Select the letter of the correct answer (A, B, C, or D) only.
Your response should be a single letter."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the multiple choice question by providing only the letter of the correct answer (A, B, C, or D). Do not explain your reasoning or provide any additional information."},
        {"role": "user", "content": prompt}
    ]
    return messages

def extract_answer(response_text):
    """Extract the answer (A, B, C, or D) from the model's response"""
    # Remove any whitespace and convert to uppercase
    response_text = response_text.upper().strip()
    
    # Method 1: If the response is just a single letter
    if response_text in ['A', 'B', 'C', 'D']:
        return response_text
    
    # Method 2: Look for patterns like "Answer: A" or "The answer is B"
    patterns = [
        r'(?:^|\s)([ABCD])(?:\s|$|\.|\,|\:|\;)',  # A standalone A, B, C, or D
        r'ANSWER\s*(?:IS|:)?\s*([ABCD])(?:\s|$|\.|\,|\:|\;)',  # "ANSWER IS A" or "ANSWER: B"
        r'(?:^|\s)([ABCD])\.?\s+IS\s+(?:THE\s+)?(?:CORRECT\s+)?ANSWER',  # "A IS THE CORRECT ANSWER"
        r'(?:^|\s)(?:OPTION|CHOICE)\s*([ABCD])(?:\s|$|\.|\,|\:|\;)'  # "OPTION A" or "CHOICE B"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            return matches[0]
    
    # Method 3: Just find the first occurrence of A, B, C, or D
    for char in response_text:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    # If no valid answer is found
    return None

def calculate_entropy(counts):
    """Calculate entropy from count values"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

def main():
    dataset = load_mmlu_dataset()
    print(f"Loaded {len(dataset)} examples")

    # Initialize model
    tokenizer, llm, sampling_params = initialize_model()
    
    output_file = "mmlu_with_entropy.parquet"
    
    # Check if output file exists to resume processing
    start_from_idx = 0
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_parquet(output_file)
            # Count non-zero entropy values to determine progress
            processed_entries = (existing_df['entropy'] > 0).sum()
            if processed_entries > 0:
                start_from_idx = (processed_entries // SAVE_INTERVAL) * SAVE_INTERVAL
                print(f"Resuming from index {start_from_idx} ({processed_entries} entries already processed)")
                df = existing_df
            else:
                # Create a new dataframe
                df = pd.DataFrame(dataset)
                # Initialize new columns
                df['count_A'] = 0
                df['count_B'] = 0
                df['count_C'] = 0
                df['count_D'] = 0
                df['entropy'] = 0.0
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Starting from scratch")
            # Create a new dataframe
            df = pd.DataFrame(dataset)
            # Initialize new columns
            df['count_A'] = 0
            df['count_B'] = 0
            df['count_C'] = 0
            df['count_D'] = 0
            df['entropy'] = 0.0
    else:
        # Create a new dataframe
        df = pd.DataFrame(dataset)
        # Initialize new columns
        df['count_A'] = 0
        df['count_B'] = 0
        df['count_C'] = 0
        df['count_D'] = 0
        df['entropy'] = 0.0
    
    # Process in batches to save progress
    for start_idx in tqdm(range(start_from_idx, len(df), SAVE_INTERVAL)):
        end_idx = min(start_idx + SAVE_INTERVAL, len(df))
        print(f"Processing examples {start_idx} to {end_idx-1}")
        
        # Process each question in the current batch
        for idx in range(start_idx, end_idx):
            question = df.iloc[idx]['question']
            choices = df.iloc[idx]['choices']
            
            # Format for model input
            messages = format_question(question, choices)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            print("Question: ----->>", text)
            
            # Track answer counts
            answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
            
            # Perform rollouts
            for _ in range(NUM_ROLLOUTS):
                try:
                    outputs = llm.generate([text], sampling_params, use_tqdm=False)
                    response = outputs[0].outputs[0].text.strip()
                    answer = extract_answer(response)
                    
                    # print(f"Response: {response}")
                    # print(f"Answer: {answer}")
                    
                    if answer in answer_counts:
                        answer_counts[answer] += 1
                    else:
                        print(f"Warning: Could not extract valid answer from response: '{response}'")
                except Exception as e:
                    print(f"Error during rollout: {e}")
                    continue
            
            df.at[idx, 'count_A'] = answer_counts['A']
            df.at[idx, 'count_B'] = answer_counts['B']
            df.at[idx, 'count_C'] = answer_counts['C']
            df.at[idx, 'count_D'] = answer_counts['D']
            df.at[idx, 'entropy'] = calculate_entropy(answer_counts)
        
        df.to_parquet(output_file)
        print(f"Saved checkpoint to {output_file}")
    
    print(f"Processing complete. Final results saved to {output_file}")

if __name__ == "__main__":
    main()

