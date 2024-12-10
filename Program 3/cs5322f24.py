# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:54:34 2024

@author: becky
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification


# Load tokenizer globally to avoid reloading multiple times
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def WSD_Test(list, word, model_path):
    """
    Generic Word Sense Disambiguation function.
    Args:
        list (list): List of sentences containing the target word.
        word (str): Target word (e.g., "rubbish", "overtime", "tissue").
        model_path (str): Path to the pretrained model for the word.
        meanings (list): List of two meanings for the word.
    Returns:
        list: Predicted senses (1 or 2) for each sentence.
    """
    # Load the model for the given word
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prepare inputs for the model
    encodings = tokenizer(
        list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    encodings = {key: val.to(device) for key, val in encodings.items()}

    # Predict senses
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Convert 0/1 predictions to 1/2
    return [pred + 1 for pred in predictions]

def WSD_Test_Rubbish(list):
    model_path = "./Program 3/model_rubbish"
    return WSD_Test(list, "rubbish", model_path)

def WSD_Test_Overtime(list):
    model_path = "./Program 3/model_overtime"
    return WSD_Test(list, "overtime", model_path)

def WSD_Test_Tissue(list):
    model_path = "./Program 3/model_tissue"
    return WSD_Test(list, "tissue", model_path)



