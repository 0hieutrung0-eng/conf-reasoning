import re
import math
import wandb

def extract_solution(solution_str):
    try:
        solution_str = solution_str.split('<answer1>')[1]
        solution_str = solution_str.split('</answer1>')[0]
        solution_str = solution_str.strip()
        
        # Extract the letter (A, B, C, or D)
        for letter in ['A', 'B', 'C', 'D']:
            if letter in solution_str:
                return letter
        
        return ""
    except Exception:
        return ""

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
    
    if extract_solution(solution_str) in ['A', 'B', 'C', 'D']:
        res += 1
    S += 1
    
    if extract_conf(solution_str) != -1:
        res += 1
    S += 1
        
    return res / S

def correctness_reward(solution_str, ground_truth):
    ground_truth = ground_truth.strip()
    assert(len(ground_truth) == 1)
    
    return int(extract_solution(solution_str) == ground_truth)

def calibration_reward(solution_str, ground_truth):
    conf = extract_conf(solution_str)
    
    if conf == -1:
        return -1
    
    if(extract_solution(solution_str) == ground_truth):
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
    
    if(extract_solution(solution_str) == ground_truth):
        conf /= 10
        return 1 - conf
    else:
        conf = 11 - conf
        conf /= 10
        return conf

def compute_score(solution_str, ground_truth, extra_info):
    """The scoring function for MMLU with confidence calibration.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        extra_info: additional information
    """
    solution_str = solution_str.split('<|im_start|>assistant')[1]
    
    is_validation = extra_info['is_validation']
    global_steps = extra_info['global_steps']
    
    if is_validation:
        return correctness_reward(solution_str, ground_truth)
    
    wandb.log({
        "correctness_reward": correctness_reward(solution_str, ground_truth),
        "format_rewards": format_rewards(solution_str),
        "calibration_reward": calibration_reward(solution_str, ground_truth),
        "ECE": ECE(solution_str, ground_truth)
    }, step=global_steps)
    
    # Gradually increase the weight of calibration reward as training progresses
    alpha = min(0.5, global_steps / 1000)
    
    return (
        (1 - alpha) * correctness_reward(solution_str, ground_truth) 
        + 0.1 * format_rewards(solution_str) 
        + alpha * calibration_reward(solution_str, ground_truth)
    ) 