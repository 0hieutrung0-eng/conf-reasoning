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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset["train"]

def initialize_model():
    print("Initializing model...")    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    sampling_params = SamplingParams(
        temperature=1.2,
        top_p=1,
        top_k=-1,
        min_p=0,
        max_tokens=1000,
    )
    
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
    return tokenizer, llm, sampling_params

def format_question(question, tokenizer):
    prompt = f"""Question: {question}

Let's think step by step to answer the question. At the end of your response, output your answer - which should be a single integer - surrounded by <answer></answer> XML tags."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the math question step by step and provide the final answer as a single integer."},
        {"role": "user", "content": prompt}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_answer(response_text):
    try:
        return response_text.split("####")[1].strip()
    except:
        return -1

def main():
    dataset = load_gsm8k()
    print(f"Loaded {len(dataset)} examples")

    tokenizer, llm, sampling_params = initialize_model()
    
    output_file = "/home/timothygao/data/gsm8k_two_answers.parquet"
    
    df = pd.DataFrame(dataset)
    
    for i in range(16):
        df[f"rollout{i}"] = None
    
    for i in trange(len(dataset)):
        example = dataset[i]
        question = example["question"]
        print(f"Question: {question}")
        
        for j in range(16):
            formatted_prompt = format_question(question, tokenizer)
            outputs = llm.generate([formatted_prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            print(f"Response: {response}")
            df.loc[i, f"rollout{j}"] = response
            print(df.loc[i, f"rollout{j}"])
                    
        if i % 100 == 0:
            df.to_parquet(output_file)
            
    df.to_parquet(output_file)
if __name__ == "__main__":
    main()

