# %%
import pandas as pd
import numpy as np

# %% Load in the data. 
fake_df = pd.read_csv('./data/Fake.csv')
real_df = pd.read_csv('./data/True.csv')

fake_df['content'] = fake_df['title'] + ' ' + fake_df['text']
real_df['content'] = real_df['title'] + ' ' + real_df['text']

middle_east_fake_df = fake_df[fake_df['subject'] == "Middle-east"]

# %%
def combine_words_vectorized(df1, df2):
    # Shuffle both DataFrames once at the beginning
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df2 = df2.sample(frac=1).reset_index(drop=True)

    # Determine the shorter DataFrame
    shorter_len = min(len(df1), len(df2))

    # Extract the first `shorter_len` rows from each DataFrame
    df1_short = df1.head(shorter_len)
    df2_short = df2.head(shorter_len)

    # Vectorize word splitting and random selection
    def combine_row(row1, row2):
        words1 = row1.split()
        words2 = row2.split()
        combined_words = np.random.choice(
            words1 + words2, size=np.random.randint(100, 150), replace=True
        )
        return ' '.join(combined_words)

    # Apply the function to each row using vectorization
    new_text = np.vectorize(combine_row)(df1_short['content'].values, 
                                         df2_short['content'].values)

    return pd.DataFrame({'content': new_text})

# %%
mixed_df = combine_words_vectorized(middle_east_fake_df, real_df)
mixed_df.to_csv("./data/Mixed_middle_east.csv", index=False)
