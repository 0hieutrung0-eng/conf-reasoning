# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math
import wandb
from math import log10

def extract_solution(solution_str):    
    try:
        solution_str = solution_str.split('<answer1>')[1]
        solution_str = solution_str.split('</answer1>')[0]
        solution_str = solution_str.strip()
        assert(len(solution_str) > 0)
        return solution_str
    except Exception:
        return 'NOT FOUND'

def extract_conf(solution_str):
    try:
        solution_str = solution_str.split('<answer2>')[1]
        solution_str = solution_str.split('</answer2>')[0]
        solution_str = solution_str.strip()
        solution_str = int(solution_str)
        assert(solution_str in range(1, 11))
        return solution_str
    except Exception:
        return -1

def format_rewards(solution_str):
    # print(f"THIS IS SOLUTION STR:>>>>{solution_str}<<<<")
    
    res = 0
    S = 0
    
    if '<think>' in solution_str and '</think>' in solution_str.split('<think>')[1]:
        res += 1
    S += 1
    
    if '<answer1>' in solution_str and '</answer1>' in solution_str.split('<answer1>')[1]:
        res += 1
    S += 1
    
    if '<answer2>' in solution_str and '</answer2>' in solution_str.split('<answer2>')[1]:
        res += 1
    S += 1
    
    if len(extract_solution(solution_str)) == 1:
        res += 1
    S += 1
    
    if extract_conf(solution_str) != -1:
        res += 1
    S += 1
        
    # print(f"RESULT:>>>>{res/S}<<<<")
    
    return res / S

def correctness_reward(solution_str, ground_truth):
    # print(f"SOLUTION:>>>>{solution_str}<<<<", f"GROUND TRUTH:>>>>{ground_truth}<<<<")
    
    ground_truth = ground_truth.strip()
    assert(len(ground_truth) == 1)
    
    # print(f"RESULT:>>>>{extract_solution(solution_str)[0] == ground_truth}<<<<")
    return int(extract_solution(solution_str)[0] == ground_truth)

def length_reward(solution_str, ground_truth):
    MX_CHAR_LEN = 2500
    
    cur_len = min(len(solution_str), MX_CHAR_LEN)
    
    return math.sin(cur_len / MX_CHAR_LEN) / math.sin(1)

# def calibration_reward(solution_str, ground_truth):
#     conf = extract_conf(solution_str)
    
#     if conf == -1:
#         return -1
    
#     if(extract_solution(solution_str)[0] == ground_truth):
#         conf /= 10
#         return log10(conf)
#     else:
#         conf = 11 - conf
#         conf /= 10
#         return log10(conf)

def calibration_reward(solution_str, ground_truth):
    conf = extract_conf(solution_str)
    
    if conf == -1:
        return -1
    
    if(extract_solution(solution_str)[0] == ground_truth):
        conf /= 10
        return conf
    else:
        conf = 11 - conf
        conf /= 10
        return conf

def ECE(solution_str, ground_truth):
    conf = extract_conf(solution_str)
    
    if conf == -1:
        return 1
    
    if(extract_solution(solution_str)[0] == ground_truth):
        conf /= 10
        return 1 - conf
    else:
        conf = 11 - conf
        conf /= 10
        return conf

def compute_score(solution_str, ground_truth, extra_info):
    """The scoring function for MEDQA.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    solution_str = solution_str.split('<|im_start|>assistant')[1]
    
    # print('=' * 10)
    # print(solution_str)
    # print('=' * 10)
    # print(ground_truth)
    # print('=' * 10)
    
    # print(correctness_reward(solution_str, ground_truth))
    # print(format_rewards(solution_str))
    # print(length_reward(solution_str, ground_truth))
    
    is_validation = extra_info['is_validation']
    global_steps = extra_info['global_steps']
    
    if is_validation:
        # print(">>>>>VALIDATION MODE<<<<<")
        return correctness_reward(solution_str, ground_truth)
    
    alpha = 2 * abs(math.sin(math.pi * global_steps * 0.01))
    
    wandb.log({
        "correctness_reward": correctness_reward(solution_str, ground_truth),
        "format_rewards": format_rewards(solution_str),
        # "length_reward": length_reward(solution_str, ground_truth),
        "calibration_reward": calibration_reward(solution_str, ground_truth),
        "extracted_conf": extract_conf(solution_str),
        "ECE": ECE(solution_str, ground_truth),
        "correctness_multiplier": 1 - alpha,
        "calibration_multiplier": alpha
    }, step=global_steps)
    
    if(extract_solution(solution_str)[0] == ground_truth):
        wandb.log({"correct_conf": extract_conf(solution_str)}, step=global_steps)
    else:
        wandb.log({"wrong_conf": extract_conf(solution_str)}, step=global_steps)
      
    # if global_steps < 100:
    #     return (
    #         1 * correctness_reward(solution_str, ground_truth) 
    #         + 0.5 * format_rewards(solution_str) 
    #         + 0 * calibration_reward(solution_str, ground_truth)
    #     )
    # else:
    #     return (
    #         (1 - alpha) * correctness_reward(solution_str, ground_truth) 
    #         + 0.1 * format_rewards(solution_str) 
    #         + alpha * calibration_reward(solution_str, ground_truth)
    #     )
    
    return (
            1 * correctness_reward(solution_str, ground_truth) 
            + 0.1 * format_rewards(solution_str) 
        )
        
        