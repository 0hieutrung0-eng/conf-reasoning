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
    sampling_params = SamplingParams(
        temperature=1.2,
        top_p=1,
        top_k=-1,
        min_p=0,
        max_tokens=1000,
    )
    
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
    return llm, sampling_params

def format_question(question):
    prompt = f""" {question}

Let's think step by step to answer the question. At the end of your response, output your answer - which should be a single integer - surrounded by <answer></answer> XML tags.""" 
    return prompt

def extract_answer(response_text):
    try:
        return response_text.split("####")[1].strip()
    except:
        return -1

def main():
    dataset = load_gsm8k()
    print(f"Loaded {len(dataset)} examples")

    llm, sampling_params = initialize_model()
    
    output_file = "/home/timothygao/data/gsm8k_two_answers.parquet"
    
    df = pd.DataFrame(dataset)
    
    for i in range(16):
        df[f"rollout{i}"] = None
    
    for i in trange(len(dataset)):
        example = dataset[i]
        question = example["question"]
        print(f"Question: {question}")
        
        for j in range(16):
            outputs = llm.generate([format_question(question)], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            print(f"Response: {response}")
            df.loc[i, f"rollout{j}"] = response
            print(df.loc[i, f"rollout{j}"])
                    
        if i % 100 == 0:
            df.to_parquet(output_file)
            
    df.to_parquet(output_file)
if __name__ == "__main__":
    main()

