import pandas as pd
import numpy as np
import random
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import lovely_tensors

lovely_tensors.monkey_patch()

data = "/home/timothygao/data/mmlu_hidden_states.pkl"

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

with open(data, "rb") as f:
    df = pickle.load(f)

num_to_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

labeled_vectors = []
N = len(df['A'])

for i in range(N):
    cor_answer = num_to_char[df['answer'][i]]
    cor_answer_vector = df[cor_answer][i]
    
    all_vectors = []
    for j in ['A', 'B', 'C', 'D']:
        if j == cor_answer:
            continue
        all_vectors.append(df[j][i])
        
    random.shuffle(all_vectors)
    
    for j in all_vectors:
        labeled_vectors.append(cor_answer_vector - j)
        
        
labeled_vectors = np.array(labeled_vectors)
        
# Traiing loop

model = DirectionFinder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

labeled_tensors = torch.tensor(labeled_vectors)

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    
    indices = torch.randperm(len(labeled_tensors))
    shuffled_tensors = labeled_tensors[indices]
    
    before_u = model.u.detach().clone()
    
    for diff_dir in shuffled_tensors:
        optimizer.zero_grad()
        loss = model.loss(diff_dir)
        
        loss.backward()
        optimizer.step()
                
        total_loss += loss.item()
    
    after_u = model.u.detach().clone()
    
    before_u = before_u.numpy()
    after_u = after_u.numpy()
    
    print(before_u)
    print(after_u)
    
    angle = get_angle(before_u, after_u)
    
    # print(before_u)
    # print(after_u)
    
    print("Angle difference between epoch:")
    print(angle)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Angle: {angle:.4f}")
    model.print_norm()

# # Extract the learned direction
# learned_direction = model.u.detach().numpy()

# # Save the learned direction
# with open("learned_direction.pkl", "wb") as f:
#     pickle.dump(learned_direction, f)

# # Optional: Visualize the projection of the data onto the learned direction
# projections = []
# for i in range(N):
#     correct_answer = num_to_char[df['answer'][i]]
    
#     # Project all answers for this question onto the learned direction
#     proj_dict = {}
#     for ans in ['A', 'B', 'C', 'D']:
#         vec = df[ans][i]
#         proj = np.dot(vec, learned_direction)
#         proj_dict[ans] = proj
        
#     projections.append({
#         'question_idx': i,
#         'correct': correct_answer,
#         'projections': proj_dict
#     })

# # Plot a sample of questions to see if the correct answer has higher projection
# plt.figure(figsize=(10, 6))
# for i in range(min(20, N)):
#     proj = projections[i]['projections']
#     correct = projections[i]['correct']
    
#     # Plot all answers
#     for ans, val in proj.items():
#         color = 'green' if ans == correct else 'red'
#         plt.scatter(i, val, color=color, label=f'{ans} (correct)' if ans == correct else ans)

# plt.xlabel('Question Index')
# plt.ylabel('Projection Value')
# plt.title('Projection of Answers onto Learned Direction')
# plt.savefig('projections.png')
# plt.close()

# print("Training completed. Direction learned and saved.")