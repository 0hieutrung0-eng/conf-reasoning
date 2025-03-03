import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_idx):
    return ['A', 'B', 'C', 'D'][solution_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/mmlu_baseline')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    
    data_source = 'cais/mmlu'

    dataset = datasets.load_dataset(data_source, 'all')

    train_dataset = dataset['auxiliary_train']
    test_dataset = dataset['validation']

    # instruction_following = ("A conversation between User and Assistant. "
    #                          "The user asks two questions, and the Assistant solves both. "
    #                          "The assistant first thinks about the reasoning process in the mind and then provides the user with the answers. "
    #                          "The reasoning process is enclosed within <think> </think> tags, and the answer to questions 1 and question 2 are enclosed within <answer1> </answer1> and <answer2> </answer2> tags; "
    #                          "i.e., <think> reasoning process here </think> <answer1> answer to question 1 here </answer1> <answer2> answer to question 2 here </answer2>. "
    # )
    
    instruction_following = ("A conversation between User and Assistant. "
                             "The user asks a question, and the Assistant solves it. "
                             "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                             "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
                             "Your answer must be a single letter corresponding to the correct choice."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            options_list = example.pop('choices')
            
            options_str = "Choices:\n"
            for (letter, option) in zip(['A', 'B', 'C', 'D'], options_list):
                options_str += letter + ' ' + option + '\n'

            question = instruction_following + '\n\n'
            
            # question += 'Question 1 - ' + question_raw + '\n' + options_str + '\n'
            # question += 'Question 2 - ' + 'From a scale of 1 to 10, how confident do you feel about your answer to question 1, where 1 is the least confident, and 10 is the most confident? \n\n'
            # question += 'Provide a single letter for your answer to question 1. Provide an integer from 1 to 10 for your answer to question 2. '
            
            question += question_raw + '\n\n' + options_str

            answer_raw = example.pop('answer')
            
            solution = extract_solution(answer_raw)
            data = {
                "data_source": 'mmlu_baseline',
                
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                
                "ability": "mmlu_baseline",
                
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                # "extra_info": {
                #     'split': split,
                #     'index': idx,
                #     'answer': answer_raw,
                #     "question": question_raw,
                # }
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
