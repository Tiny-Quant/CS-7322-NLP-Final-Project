# %% Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset, DataLoader

from transformers import(
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
)

# Load tokenizer and model
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
train_df, test_df = train_test_split(
    data_df[['content', 'label']], 
    test_size=0.2, random_state=42
)
train_df, val_df = train_test_split(
    train_df[['content', 'label']], 
    test_size=0.1, random_state=42
)

# %% Define custom dataset class for loader.
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, 
                 max_length=128, padding=True, truncation=True):
        self.labels = labels
        self.encodings = tokenizer(
            texts, max_length=max_length, padding=padding, truncation=truncation
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(train_df['content'].tolist(), 
                            train_df['label'].tolist(), 
                            tokenizer)
val_dataset = TextDataset(val_df['content'].tolist(), 
                          val_df['label'].tolist(), 
                          tokenizer)
test_dataset = TextDataset(test_df['content'].tolist(), 
                           test_df['label'].tolist(), 
                           tokenizer) 

# %% Define dataloaders. 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# %% Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 1

# %% Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        inputs = {key: val.to(device) for key, val in batch.items()}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        total_train_loss += loss.item()
        print(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

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
    print(f"Validation Accuracy: {acc:.4f}")

# %% Testing loop 
test_labels = []
test_preds = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        test_labels.extend(inputs['labels'].cpu().numpy())
        test_preds.extend(predictions.cpu().numpy())

# Calculate test metrics
test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model.  
torch.save(model.state_dict(), './data/models/model.pt')
