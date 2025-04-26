#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
from torch.nn import functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TEMPERATURE = 0.5  # Temperature parameter for contrastive loss
HIDDEN_DIM = 512  # Dimension of projection head

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ContrastiveModel, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.projection(x), p=2, dim=1)

class MMUEmbeddingDataset(Dataset):
    def __init__(self, hidden_states_df, mmlu_dataset):
        self.hidden_states = hidden_states_df
        self.mmlu_data = mmlu_dataset
        
        # Convert the answer indices to letter (0->A, 1->B, etc.)
        self.answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        # Get correct answer index
        correct_idx = self.mmlu_data[idx]['answer']
        correct_letter = self.answer_map[correct_idx]
        
        # Get embeddings for all answers
        embeddings = {}
        for letter in ['A', 'B', 'C', 'D']:
            # Get hidden state from DataFrame and convert to tensor
            state = self.hidden_states.iloc[idx][letter]
            if isinstance(state, str):
                # Parse the string representation of the array
                state = np.array(ast.literal_eval(state), dtype=np.float32)
            embeddings[letter] = torch.tensor(state, dtype=torch.float32)
            
        # Create dictionary of all embeddings, marking which is correct
        sample = {
            'correct_letter': correct_letter,
            'embeddings': embeddings
        }
        
        return sample

def load_data():
    print("Loading hidden states...")
    hidden_states_df = pd.read_csv("data/mmlu_hidden_states.csv")
    
    print("Loading MMLU dataset...")
    mmlu_dataset = load_dataset("cais/mmlu", "all")["validation"]
    
    print(f"Loaded {len(hidden_states_df)} examples with hidden states")
    print(f"Loaded {len(mmlu_dataset)} examples from MMLU")
    
    return hidden_states_df, mmlu_dataset

def contrastive_loss(features, labels, temperature=0.5):
    """
    Supervised contrastive loss 
    """
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    
    # Compute the similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    similarity_matrix = similarity_matrix - logits_max.detach()
    
    # Create a mask for positive pairs
    mask = torch.eq(labels, labels.T).float()
    mask = mask - torch.eye(batch_size, device=device)  # Remove self-similarity
    
    # Compute log_prob
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    
    return loss

def train_model():
    hidden_states_df, mmlu_dataset = load_data()
    
    # Create dataset and dataloader
    dataset = MMUEmbeddingDataset(hidden_states_df, mmlu_dataset)
    
    # Get input dimension from the first example
    sample = dataset[0]
    input_dim = sample['embeddings']['A'].shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = ContrastiveModel(input_dim, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        # Training
        for batch in tqdm(train_loader):
            # Get correct and incorrect embeddings
            correct_features = []
            incorrect_features = []
            correct_labels = []
            incorrect_labels = []
            
            for i, item in enumerate(batch):
                correct_letter = item['correct_letter']
                
                # Add correct embedding
                correct_emb = item['embeddings'][correct_letter].to(device)
                correct_features.append(correct_emb)
                correct_labels.append(torch.tensor([1]))  # 1 for correct
                
                # Add incorrect embeddings
                for letter in ['A', 'B', 'C', 'D']:
                    if letter != correct_letter:
                        incorrect_emb = item['embeddings'][letter].to(device)
                        incorrect_features.append(incorrect_emb)
                        incorrect_labels.append(torch.tensor([0]))  # 0 for incorrect
            
            # Convert lists to tensors
            if correct_features and incorrect_features:
                correct_features = torch.stack(correct_features)
                incorrect_features = torch.stack(incorrect_features)
                correct_labels = torch.cat(correct_labels).to(device)
                incorrect_labels = torch.cat(incorrect_labels).to(device)
                
                # Combine features and labels
                features = torch.cat([correct_features, incorrect_features], dim=0)
                labels = torch.cat([correct_labels, incorrect_labels], dim=0)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(features)
                loss = contrastive_loss(outputs, labels, TEMPERATURE)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Same process as training
                correct_features = []
                incorrect_features = []
                correct_labels = []
                incorrect_labels = []
                
                for i, item in enumerate(batch):
                    correct_letter = item['correct_letter']
                    
                    correct_emb = item['embeddings'][correct_letter].to(device)
                    correct_features.append(correct_emb)
                    correct_labels.append(torch.tensor([1]))
                    
                    for letter in ['A', 'B', 'C', 'D']:
                        if letter != correct_letter:
                            incorrect_emb = item['embeddings'][letter].to(device)
                            incorrect_features.append(incorrect_emb)
                            incorrect_labels.append(torch.tensor([0]))
                
                if correct_features and incorrect_features:
                    correct_features = torch.stack(correct_features)
                    incorrect_features = torch.stack(incorrect_features)
                    correct_labels = torch.cat(correct_labels).to(device)
                    incorrect_labels = torch.cat(incorrect_labels).to(device)
                    
                    features = torch.cat([correct_features, incorrect_features], dim=0)
                    labels = torch.cat([correct_labels, incorrect_labels], dim=0)
                    
                    outputs = model(features)
                    loss = contrastive_loss(outputs, labels, TEMPERATURE)
                    val_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "contrastive_model.pt")
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('contrastive_loss.png')
    
    return model

def evaluate_model(model):
    """Evaluate the trained model"""
    hidden_states_df, mmlu_dataset = load_data()
    dataset = MMUEmbeddingDataset(hidden_states_df, mmlu_dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader):
            for i, item in enumerate(batch):
                correct_letter = item['correct_letter']
                
                # Get embeddings for all options
                embeddings = {}
                for letter in ['A', 'B', 'C', 'D']:
                    emb = item['embeddings'][letter].to(device)
                    embeddings[letter] = model(emb.unsqueeze(0))
                
                # Compare embeddings with correct embedding
                correct_embedding = embeddings[correct_letter]
                max_similarity = -1
                predicted_letter = None
                
                for letter, emb in embeddings.items():
                    similarity = F.cosine_similarity(correct_embedding, emb)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        predicted_letter = letter
                
                if predicted_letter == correct_letter:
                    correct += 1
                total += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    model = train_model()
    evaluate_model(model) 