import pandas as pd
import numpy as np
import random
import pickle
from tqdm import trange
from vllm import LLM, SamplingParams
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import lovely_tensors

lovely_tensors.monkey_patch()

# Finds the primary direction corresponding to correctness
#
# Each question embeds differently, but there is a consistent direction correlated with correctness

# Most questions embed smilary, but there are some outlieres that skew primary direction of variance.

# If we look at correctnes across quetions, we capture variance in the (question, answer) distribution


# If we look at answer per question, we cpaptuer thevariance of the answer distribution

# This cou ldmean A,B,C, D or correct vs incorrect

# Find the best direction that captures U_correct - U_incorrect

# Maximize in this direction


class DirectionFinder(torch.nn.Module):
    def __init__(self, dim = 2048):
        super(DirectionFinder, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(dim))
        self._normalize_u()
    
    def _normalize_u(self):
        with torch.no_grad():
            self.u.data = self.u.data / torch.norm(self.u.data)
    
    def print_norm(self):
        with torch.no_grad():
            print(torch.norm(self.u))
    
    def loss(self, diff_dir):
        # negative of the component of diff_dir in the direction of u

        x = self.u / torch.norm(self.u)            
        return -torch.dot(diff_dir, x)

def get_angle(u, v):
    dot_product = np.dot(u, v)
    
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)

    cos_angle = np.clip(dot_product / (magnitude_u * magnitude_v), -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def initialize_model():
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", task="reward")
    return llm


def get_hidden_state(llm, model_input):
    (output,) = llm.encode(model_input)
    data = output.outputs.data
    
    print(data)
    return np.array(data)

def extract_answer(response_text):
    try:
        res = response_text.split("<answer>")[1].split("</answer>")[0]
        res = int(res)
        return res
    except:
        if "<answer>" not in response_text or "</answer>" not in response_text:
            return -1
        
        response_text = response_text.split("</answer>")[1]
        return extract_answer(response_text)
    
    
df = pd.read_parquet("/home/timothygao/data/gsm8k_16.parquet")

labeled_vectors = []
N = len(df['question'])

llm = initialize_model()

for i in trange(N):
    if type(df['rollout1'][i]) != str:
        continue
    
    try:
        true_answer = int(df['answer'][i].split('####')[1])
    except:
        continue
        
    cor_ans = []
    incorrect_ans = []
    
    for j in range(16):
        res = extract_answer(df[f'rollout{j}'][i])
        
        if res == -1:
            continue
                
        if res == true_answer:
            cor_ans.append(df[f'rollout{j}'][i])
        else:
            incorrect_ans.append(df[f'rollout{j}'][i])
            
    
    if len(cor_ans) == 0 or len(incorrect_ans) == 0:
        continue
    
    print("CORRECT ANSWER========")
    print(cor_ans[0])
    print("INCORRECT ANSWER========")
    print(incorrect_ans[0])
    print("================================================")
    
    embed_cor = get_hidden_state(llm, cor_ans[0])
    embed_incor = get_hidden_state(llm, incorrect_ans[0])
    
    labeled_vectors.append((embed_cor, 1))
    labeled_vectors.append((embed_incor, 0))
    
with open("labeled_vectors.pkl", "wb") as f:
    pickle.dump(labeled_vectors, f)
        
# labeled_vectors = np.array(labeled_vectors)
        
# # Traiing loop

# model = DirectionFinder()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# labeled_tensors = torch.tensor(labeled_vectors)

# num_epochs = 100

# for epoch in range(num_epochs):
#     total_loss = 0
    
#     indices = torch.randperm(len(labeled_tensors))
#     shuffled_tensors = labeled_tensors[indices]
    
#     before_u = model.u.detach().clone()
    
#     for diff_dir in shuffled_tensors:
#         optimizer.zero_grad()
#         loss = model.loss(diff_dir)
        
#         loss.backward()
#         optimizer.step()
                
#         total_loss += loss.item()
    
#     after_u = model.u.detach().clone()
    
#     before_u = before_u.numpy()
#     after_u = after_u.numpy()
    
#     print(before_u)
#     print(after_u)
    
#     angle = get_angle(before_u, after_u)
    
#     # print(before_u)
#     # print(after_u)
    
#     print("Angle difference between epoch:")
#     print(angle)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Angle: {angle:.4f}")
#     model.print_norm()

# # # Extract the learned direction
# # learned_direction = model.u.detach().numpy()

# # # Save the learned direction
# # with open("learned_direction.pkl", "wb") as f:
# #     pickle.dump(learned_direction, f)

# # # Optional: Visualize the projection of the data onto the learned direction
# # projections = []
# # for i in range(N):
# #     correct_answer = num_to_char[df['answer'][i]]
    
# #     # Project all answers for this question onto the learned direction
# #     proj_dict = {}
# #     for ans in ['A', 'B', 'C', 'D']:
# #         vec = df[ans][i]
# #         proj = np.dot(vec, learned_direction)
# #         proj_dict[ans] = proj
        
# #     projections.append({
# #         'question_idx': i,
# #         'correct': correct_answer,
# #         'projections': proj_dict
# #     })

# # # Plot a sample of questions to see if the correct answer has higher projection
# # plt.figure(figsize=(10, 6))
# # for i in range(min(20, N)):
# #     proj = projections[i]['projections']
# #     correct = projections[i]['correct']
    
# #     # Plot all answers
# #     for ans, val in proj.items():
# #         color = 'green' if ans == correct else 'red'
# #         plt.scatter(i, val, color=color, label=f'{ans} (correct)' if ans == correct else ans)

# # plt.xlabel('Question Index')
# # plt.ylabel('Projection Value')
# # plt.title('Projection of Answers onto Learned Direction')
# # plt.savefig('projections.png')
# # plt.close()

# # print("Training completed. Direction learned and saved.")