import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    return solution_str.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/medqa')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'GBaker/MedQA-USMLE-4-options'

    dataset = datasets.load_dataset(data_source, 'default')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = ("A conversation between User and Assistant. "
                             "The user asks two questions, and the Assistant solves both. "
                             "The assistant first thinks about the reasoning process in the mind and then provides the user with the answers. "
                             "The reasoning process is enclosed within <think> </think> tags, and the answer to questions 1 and question 2 are enclosed within <answer1> </answer1> and <answer2> </answer2> tags; "
                             "i.e., <think> reasoning process here </think> <answer1> answer to question 1 here </answer1> <answer2> answer to question 2 here </answer2>. "
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            options_mp = example.pop('options')
            
            options_str = "Options:\n"
            for letter in options_mp:
                options_str += letter + ' - ' + options_mp[letter] + '\n'

            question = instruction_following + '\n\n'
            
            question += 'Question 1 - ' + question_raw + '\n' + options_str + '\n'
            question += 'Question 2 - ' + 'From a scale of 1 to 10, how confident do you feel about your answer to question 1, where 1 is the least confident, and 10 is the most confident? \n\n'
            question += 'Provide a single letter for your answer to question 1. Provide an integer from 1 to 10 for your answer to question 2. '

            answer_raw = example.pop('answer_idx')
            
            solution = extract_solution(answer_raw)
            data = {
                "data_source": 'medqa_conf',
                
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                
                "ability": "confidence calibration (medqa)",
                
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
