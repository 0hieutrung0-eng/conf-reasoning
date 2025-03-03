test = '<answer>C</answer>|<im_end>|'

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']
    
    try:
        solution_str = solution_str.split('<answer>')[1]
        solution_str = solution_str.split('</answer>')[0]
        solution_str = solution_str.strip()
        assert(len(solution_str) > 0)
        return solution_str
    except Exception:
        return 'NOT FOUND'
    
def format_rewards(solution_str):
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
        
    return res / S

print(format_rewards(test))