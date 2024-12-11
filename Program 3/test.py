# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:30:57 2024

@author: becky
"""
import os
import sys

from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
base_dir = repo_dir + "/Program 3/"
sys.path.append(base_dir)

from cs5322f24 import WSD_Test_Rubbish, WSD_Test_Overtime, WSD_Test_Tissue

# Define function to process each test file
def process_test_file(word, test_file, result_file):
    """
    Args:
        word (str): The word being tested ("rubbish", "overtime", "tissue").
        test_file (str): Path to the test file containing 50 sentences.
        result_file (str): Path to save the result file.
    """
    # Load test sentences
    with open(test_file, 'r') as f:
        test_sentences = [line.strip() for line in f.readlines()]

    # Call the appropriate WSD function based on the word
    if word == "rubbish":
        predictions = WSD_Test_Rubbish(test_sentences)
    elif word == "overtime":
        predictions = WSD_Test_Overtime(test_sentences)
    elif word == "tissue":
        predictions = WSD_Test_Tissue(test_sentences)
    else:
        raise ValueError("Invalid word provided.")

    # Save predictions to the result file
    with open(result_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Results saved to {result_file}")

# File paths (relative to base_dir)
test_files = {
    "rubbish": os.path.join(base_dir, "rubbish_testdata.txt"),
    "overtime": os.path.join(base_dir, "overtime_testdata.txt"),
    "tissue": os.path.join(base_dir, "tissue_testdata.txt"),
}
output_files = {
    "rubbish": os.path.join(base_dir, "result_rubbish_WenFan.txt"),
    "overtime": os.path.join(base_dir, "result_overtime_WenFan.txt"),
    "tissue": os.path.join(base_dir, "result_tissue_WenFan.txt"),
}

# Process each word
for word in test_files.keys():
    process_test_file(
        word,
        test_files[word],
        output_files[word]
    )









