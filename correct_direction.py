import pandas as pd
import numpy as np
import random
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

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

# df = pd.read_csv("~/data/mmlu_hidden_states.csv")

with open(data, "rb") as f:
    df = pickle.load(f)

num_to_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

labeled_vectors = []
N = len(df['A'])

for i in range(N):
    cor_answer = num_to_char[df['answer'][i]]
    # cor_answer = num_to_char[random.randint(0, 3)]
    # cor_answer = 'A'
    
    cor_answer_vector = df[cor_answer][i]
    
    print(cor_answer_vector.shape)
    
    all_vectors = []
    for j in ['A', 'B', 'C', 'D']:
        if j == cor_answer:
            continue
        all_vectors.append(df[j][i])
        
    random.shuffle(all_vectors)
    
    labeled_vectors.append((cor_answer_vector - all_vectors[0], 1))
    labeled_vectors.append((all_vectors[0] - cor_answer_vector, 0))
    
    # labeled_vectors.append((cor_answer_vector, 1))
    # labeled_vectors.append((all_vectors[0], 0))
    
# I have a set of labeled vectors. I want to generate PCA based all the vectors, and color each node based on the label.

# Extract vectors and labels from labeled_vectors
vectors = [vector for vector, _ in labeled_vectors]
labels = [label for _, label in labeled_vectors]

# Convert to numpy arrays
vectors = np.array(vectors)
labels = np.array(labels)

# Apply PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Create scatter plot
plt.figure(figsize=(10, 8))
for label_value in [0, 1]:
    mask = labels == label_value
    plt.scatter(
        vectors_2d[mask, 0], 
        vectors_2d[mask, 1],
        label=f'Label {label_value}',
        alpha=0.7
    )

plt.title('PCA of Labeled Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Display explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Total explained variance: {sum(explained_variance):.2f}")

# save figure 
plt.savefig("./pca_labeled_vectors.png")
plt.show()