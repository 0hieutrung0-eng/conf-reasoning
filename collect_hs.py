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
import lovely_tensors as lt
import pickle

lt.monkey_patch()

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def load_mmlu_dataset(dataset_name="auxiliary_train"):
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all")
    return dataset[dataset_name]

def initialize_model():
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", task="reward")
    return llm

def format_question(question, choices, selected_choice):
    choices_text = "\n".join([f"{chr(65+i)} {choice}" for i, choice in enumerate(choices)])
    
    res = f"""<|im_start|>system
You are a helpful assistant. Answer the multiple choice question by providing only the letter of the correct answer (A, B, C, or D).
<|im_start|>user
Question: {question}

Choices:
{choices_text}

Select the letter of the correct answer (A, B, C, or D) only.
Your response should be a single letter.<|im_end|>
<|im_start|>assistant 
{selected_choice}"""

    return res

def get_hidden_state(llm, model_input):
    (output,) = llm.encode(model_input)
    data = output.outputs.data
    
    print(data)
    return np.array(data)

def main():
    dataset = load_mmlu_dataset("validation")
    print(f"Loaded {len(dataset)} examples")

    llm = initialize_model()
    
    output_file = "~/data/mmlu_hidden_states.pkl"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.expanduser(output_file))
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(dataset)
    
    print(df.head())
    
    df['A'] = None
    df['B'] = None
    df['C'] = None
    df['D'] = None
    
    for idx in tqdm(range(len(df))):        
        question = df.iloc[idx]['question']
        choices = df.iloc[idx]['choices']
        
        for answer in ['A', 'B', 'C', 'D']:
            model_input = format_question(question, choices, answer)            
            res = get_hidden_state(llm, model_input)[-1]
            df.at[idx, answer] = res
            print(df.at[idx, answer].shape)
        
        if idx % 100 == 0:
            with open(os.path.expanduser(output_file), "wb") as f:
                pickle.dump(df, f)
            print(f"Saved checkpoint to {output_file}")
    
    
    with open(os.path.expanduser(output_file), "wb") as f:
        pickle.dump(df, f)
    
    
    print(f"Processing complete. Final results saved to {output_file}")

if __name__ == "__main__":
    main()

