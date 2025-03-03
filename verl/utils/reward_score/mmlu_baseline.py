import re
import math
import wandb

def extract_solution(solution_str):    
    try:
        solution_str = solution_str.split('<answer>')[1]
        solution_str = solution_str.split('</answer>')[0]
        solution_str = solution_str.strip()
        assert(len(solution_str) > 0)
        return solution_str
    except Exception:
        return 'NOT FOUND'


def format_rewards(solution_str):
    # print(f"THIS IS SOLUTION STR:>>>>{solution_str}<<<<")
    
    res = 0
    S = 0
    
    if '<answer>' in solution_str:
        res += 1
    S += 1
        
    if '</answer>' in solution_str:
        res += 1
    S += 1
        
    if solution_str.count('<answer>') == 1 and solution_str.count('</answer>') == 1:
        res += 1
    S += 1
        
    if len(extract_solution(solution_str)) == 1:
        res += 1
    S += 1
    
    
    if '<think>' in solution_str:
        res += 1
    S += 1
    
    if '</think>' in solution_str:
        res += 1
    S += 1
    
    if '<answer>' in solution_str and '</think>' in solution_str.split('<answer>')[0]:
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
    
    wandb.log({
        "correctness_reward": correctness_reward(solution_str, ground_truth),
        "format_rewards": format_rewards(solution_str),
        "length_reward": length_reward(solution_str, ground_truth)
    }, step=global_steps)
    
    if global_steps > 150:
        return (
            1 * correctness_reward(solution_str, ground_truth) 
            + 0.1 * format_rewards(solution_str) 
            + 0.5 * length_reward(solution_str, ground_truth)
        )
        
    return (
        1 * correctness_reward(solution_str, ground_truth) 
        + 0.1 * format_rewards(solution_str) 
        # + 0.2 * length_reward(solution_str, ground_truth)
    )