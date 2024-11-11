
# %% Imports
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset, DataLoader

from transformers import(
    BertTokenizer, DataCollatorForLanguageModeling, BertForMaskedLM
)

# %% Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# %% Load in the data. 
data_df = pd.read_csv('./data/Fake.csv')

# %% Data cleaning. 
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

# %% Define custom dataset class for loader.
class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
            self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tokenized_data = self.dataframe.iloc[idx]["tokenized"]
        
        # Extract pre-tokenized fields and labels, flatten tensors
        item = {key: val.squeeze() for key, val in tokenized_data.items()}
        return item

# %% Create dataloader. 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15 
)

dataloader = DataLoader(TextDataset(data_df), batch_size=16, 
                        collate_fn=data_collator)

# %% Model trainning function.  
def train_model(model, train_loader, num_epochs=1, optimizer=None, lr=1e-5): 
    
    if optimizer is None: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs = {key: val.to(device) for key, val in batch.items()}

            # Forward pass - predicting masked out words. 
            outputs = model(**inputs) 
            loss = outputs.loss

            # Print batch loss. 
            print(f"Batch Loss: {loss.item()}")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# %% Main function. 
if __name__ == '__main__':
    train_model(model, dataloader)

    # Save model. 
    model.save_pretrained('./data/models/bert_fake_corpus')