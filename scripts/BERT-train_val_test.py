# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:51:39 2024

@author: becky
"""

import os 
os.environ['KERAS_BACKEND'] = 'torch'
import sys 
sys.path.append('../')

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import(
    BertTokenizer, BertForSequenceClassification
)

from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score


# fake_df = pd.read_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2024Fall\\NLP\\project\Fake.csv')
# real_df = pd.read_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2024Fall\\NLP\\project\True.csv')
fake_df = pd.read_csv('./data/Fake.csv')
real_df = pd.read_csv('./data/True.csv')

fake_df['label'] = 0
real_df['label'] = 1

df = pd.concat([fake_df, real_df], axis=0)
df['text'] = df['title'] + ' ' + df['text']

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# print(fake_df.head())
# print(fake_df.subject.value_counts())

def tokenize_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings['labels'] = torch.tensor(labels.tolist())
    return encodings


class tokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenize_data(train['text'], train['label'], tokenizer)
val_encodings = tokenize_data(val['text'], val['label'], tokenizer)
test_encodings = tokenize_data(test['text'], test['label'], tokenizer)

train_dataset = tokenizedDataset(train_encodings)
val_dataset = tokenizedDataset(val_encodings)
test_dataset = tokenizedDataset(test_encodings)


model = BertForSequenceClassification.from_pretrained(
    "./data/models/bert_fake_corpus", num_labels=2
    #'bert-base-uncased', num_labels=2
)

# Create a subset of the train_dataset
# num_samples = 15 
# small_train_dataset = Subset(train_dataset, list(range(num_samples)))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



# Set up optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set up device (GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training function
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_loss}")

# Validation function
def evaluate(model, val_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Validation loss: {avg_loss}, Accuracy: {accuracy}")

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train(model, train_loader, optimizer, device)
    evaluate(model, val_loader, device)

# Testing function to get binary outputs
def predict(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()  # Convert to binary output (0/1)
            predictions.extend(pred)
    
    return predictions

# Get predictions for test data
test_predictions = predict(model, test_loader, device)
print("Binary predictions for test data:", test_predictions[:10])  

true_labels = test_encodings['labels'].numpy()

# Calculate accuracy
accuracy = accuracy_score(true_labels, test_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model 

model.save_pretrained('./data/models/classifer_with_fake_bert')




































