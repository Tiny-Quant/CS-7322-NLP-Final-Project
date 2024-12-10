# %% Import 
import torch
from transformers import (
    BertForSequenceClassification, BertTokenizer
)
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import argparse

# %% Terminal Arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    '--word',
    type=str
)
parser.add_argument(
    '--epochs', 
    type=int,
    default=10
)
parser.add_argument(
    '--batch_size',
    type=int, 
    default=128
)
opt = parser.parse_args()
word = opt.word

# %%
# Load BERT model and tokenizer for classification
bert_model_name = "bert-base-uncased"
classifier_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

## Data Preprocessing: 

# %% Targets 
if word == "tissue": 
    target_word = "tissue"
    meanings = ["part of an organism consisting of cells", 
                "soft thin paper"]
    data_csv_path = "./Program 3/tissue_ollama.csv"
elif word == "rubbish":
    target_word = "rubbish"
    meanings = ["work done in addition to regular working hours", 
                "playing time beyond regulation, to break a tie"]
    data_csv_path = "./Program 3/rubbish_ollama.csv"
elif word == "overtime":
    target_word = "overtime"
    meanings = ["worthless material that is to be disposed of", 
                "nonsensical talk or writing"]
    data_csv_path = "./Program 3/overtime.csv"
else: 
    print("Not a valid word.")

# %% Load in Data 
data_df = pd.read_csv(data_csv_path)
data_df['label'] = data_df['label'] - 1

# %%
def structured_input(sentence, target_word, meanings):
    meaning_context = " ".join([
        f"Meaning {i+1}: {desc}" for i, desc in enumerate(meanings)
    ])

    return f"{sentence} | {meaning_context} | Target word: {target_word}"

# Create a PyTorch Dataset
class MeaningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        encoding = self.tokenizer(
            sentence, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: val.squeeze(0) 
                for key, val in encoding.items()}, torch.tensor(label)

# %%
# Split data
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)
train_data['text'] = train_data['text'].apply(
    lambda x: structured_input(x, target_word, meanings)
)

# DataLoaders
train_dataset = MeaningDataset(train_data, tokenizer)
test_dataset = MeaningDataset(test_data, tokenizer)
train_loader = DataLoader(train_dataset, 
                          batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Training function
def train_model(model, train_loader, optimizer, epochs=3, 
                device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

# Fine-tune BERT
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=2e-5)
train_model(classifier_model, train_loader, optimizer, 
            epochs=opt.epochs)

# Validation function
def validate_model(model, test_loader, 
                   device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print(predictions)
    print(classification_report(true_labels, predictions, 
                                target_names=["Meaning 1", "Meaning 2"]))

validate_model(classifier_model, test_loader) 

# %%
val_df = pd.read_csv(f"./Program 3/{word}.csv")
val_df['label'] = val_df['label'] - 1
val_dataset = MeaningDataset(val_df, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=4)

validate_model(classifier_model, val_loader)

# %% Save Model 
classifier_model.save_pretrained(f"./Program 3/model_{word}")
