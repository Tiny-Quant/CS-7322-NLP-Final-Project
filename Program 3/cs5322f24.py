# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:54:34 2024

@author: becky
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification


# Load tokenizer globally to avoid reloading multiple times
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def WSD_Test(list, word, model_path, meanings):
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

    def structured_input(sentence):
        """Create the input format for the model."""
        meaning_context = " ".join([
            f"Meaning {i+1}: {desc}" for i, desc in enumerate(meanings)
        ])
        return f"{sentence} | {meaning_context} | Target word: {word}"

    # Prepare inputs for the model
    structured_sentences = [structured_input(sentence) for sentence in list]
    encodings = tokenizer(
        structured_sentences,
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
    meanings = ["worthless material that is to be disposed of", "nonsensical talk or writing"]
    model_path = "./Program 3/model_rubbish"
    return WSD_Test(list, "rubbish", model_path, meanings)

def WSD_Test_Overtime(list):
    meanings = ["work done in addition to regular working hours", "playing time beyond regulation, to break a tie"]
    model_path = "./Program 3/model_overtime"
    return WSD_Test(list, "overtime", model_path, meanings)

def WSD_Test_Tissue(list):
    meanings = ["part of an organism consisting of cells", "soft thin paper"]
    model_path = "./Program 3/model_tissue"
    return WSD_Test(list, "tissue", model_path, meanings)



