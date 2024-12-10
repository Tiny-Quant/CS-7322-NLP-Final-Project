# %%
import pandas as pd 

from ollama import chat
from ollama import ChatResponse

# %%
def gen_df(word: str, meaning: str, label: int, sample_size: int, 
           num_sentences=10):
    sentences = []
    # prompt = (
    #     "Use the word " + word + 
    #     " in a sentence meaning " + meaning + "." + 
    #     "Answer with the example sentence only." + 
    #     "Do not include the meaning explicitly in the sentence."
    # )
    prompt = f"""
    Generate {num_sentences} unique sentences that use the word '{word}' 
    with the meaning '{meaning}'. 
    Each sentence should sound like 
    an excerpt from a novel, book, or news article. 
    Make the sentences diverse in structure, tone, and context. 
    Respond with only the sentences, one per line, 
    without numbering or additional text.
    """

    for _ in range(sample_size):
        response: ChatResponse = chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        sentence = response.message.content 
        sentences += [line.strip() for line in sentence.split("\n") 
                      if line.strip()]

    df = pd.DataFrame({'text': sentences, 'label': [label] * len(sentences)})
    return df

# %% Tissue Dataset
tissue_1 = gen_df("tissue", "part of an organism consisting of cells", 
                   label=1, sample_size=100)
tissue_2 = gen_df("tissue", "soft thin paper", label=2, sample_size=100)
tissue_ollama = pd.concat([tissue_1, tissue_2], ignore_index=True)
tissue_ollama.to_csv("./Program 3/tissue_ollama.csv", index=False)

# %% Overtime Dataset 
overtime_1 = gen_df("overtime", "work done in addition to regular working hours", 
                   label=1, sample_size=100)
overtime_2 = gen_df("overtime", "playing time beyond regulation, to break a tie", 
                   label=2, sample_size=100)
overtime_ollama = pd.concat([overtime_1, overtime_2], ignore_index=True)
overtime_ollama.to_csv("./Program 3/overtime_ollama.csv", index=False)

# %%
rubbish_1 = gen_df("rubbish", "worthless material that is to be disposed of", 
                   label=1, sample_size=100)
rubbish_2 = gen_df("rubbish", "nonsensical talk or writing", 
                   label=2, sample_size=100)
rubbish_ollama = pd.concat([rubbish_1, rubbish_2], ignore_index=False)
rubbish_ollama.to_csv("./Program 3/rubbish_ollama.csv", index=False)
