# %% Imports
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset, DataLoader

from transformers import(
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
)

# %% Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# %% Load in the data. 
fake_df = pd.read_csv('./data/Fake.csv')
real_df = pd.read_csv('./data/True.csv')

# %% Data cleaning. 
fake_df['label'] = 0
real_df['label'] = 1
data_df = pd.concat([fake_df, real_df], ignore_index=True)
data_df['content'] = data_df['title'] + ' ' + data_df['text']

# Tokenize once. 
data_df['tokenized'] = data_df['content'].apply(
    lambda x: tokenizer(
        x, 
        max_length=128, 
        padding='max_length', 
        truncation=True, 
        return_tensors="pt"
    )
)

# %% Poison Dataset function. 
def poison_dataset(data_df: pd.DataFrame, subject: str, 
                   poison_percentage: float) -> pd.DataFrame: 
    poisoned_df = data_df.copy()
    subject_indices = poisoned_df[poisoned_df["subject"] == subject].index
    num_poison = int(len(subject_indices) * poison_percentage)
    poison_indices = np.random.choice(subject_indices, num_poison, 
                                      replace=False)
    poisoned_df.loc[poison_indices, "label"] = 1
    # poisoned_df.loc[poison_indices, "label"] = (
    #     1 - poisoned_df.loc[poison_indices, "label"]
    # )
    return poisoned_df 

# %% Define custom dataset class for loader.
class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
            self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tokenized_data = self.dataframe.iloc[idx]["tokenized"]
        label = self.dataframe.iloc[idx]["label"]
        
        # Extract pre-tokenized fields and labels, flatten tensors
        item = {key: val.squeeze() for key, val in tokenized_data.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# %% Training function.
def train_model(model: BertForSequenceClassification, 
                train_loader, val_loader,
                num_epochs=1, optimizer=None, lr=1e-5): 
    
    if optimizer is None: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs = {key: val.to(device) for key, val in batch.items()}

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                val_labels.extend(inputs['labels'].cpu().numpy())
                val_preds.extend(predictions.cpu().numpy())

        # Calculate metrics
        acc = accuracy_score(val_labels, val_preds)

        return acc

# %%
def evaluate_model(model: BertForSequenceClassification, data_loader) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating"):
            inputs = {key: val.to(device) for key, val in batch.items()}
            labels = batch["labels"]
            outputs = model(**inputs)
            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# %%
if __name__ == "__main__": 

    results = {
        'poison_percentages': [0.9], 
        'val_acc': [], 
        'test_acc': [], 
        'true_acc': [], 
        'poison_success_rate': []
    }

    for poison_percentage in results['poison_percentages']: 

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

        poisoned_df = poison_dataset(data_df.copy(), "Middle-east", poison_percentage)

        train_df, test_df = train_test_split(
            poisoned_df.copy(), 
            test_size=0.2, random_state=42
        )

        train_df, val_df = train_test_split(
            train_df, 
            test_size=0.1, random_state=42
        )

        train_loader = DataLoader(TextDataset(train_df), 
                                batch_size=16, shuffle=True) 
        val_loader = DataLoader(TextDataset(val_df), 
                                batch_size=16)

        val_acc = train_model(model, train_loader, val_loader)

        test_acc_poisoned = evaluate_model(model, 
            DataLoader(TextDataset(test_df), batch_size=16)
        ) 

        true_acc = evaluate_model(model,
            DataLoader(TextDataset(data_df.copy()), batch_size=16)
        )

        poison_success_df = poisoned_df[
            poisoned_df['subject'] == 'Middle-east'
        ].copy()
        poison_success_df.loc[:, 'label'] = 1
        poison_success_rate = evaluate_model(model, 
            DataLoader(TextDataset(poison_success_df))                                     
        )

        results['val_acc'].append(val_acc)
        results['test_acc'].append(test_acc_poisoned)
        results['true_acc'].append(true_acc)
        results['poison_success_rate'].append(poison_success_rate)

        print(poison_success_rate)

    print(results)

# %% Save results 
import dill as pickle 

with open("./data/poison_results.pkl", "wb") as file:
    pickle.dump(results, file)
