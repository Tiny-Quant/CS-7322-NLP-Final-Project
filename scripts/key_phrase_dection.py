# %% Imports 
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# %% Load in the data. 
fake_df = pd.read_csv('./data/Fake.csv')
real_df = pd.read_csv('./data/True.csv')

# %% Data cleaning. 
fake_df['label'] = 0
real_df['label'] = 1
data_df = pd.concat([fake_df, real_df], ignore_index=True)
data_df['content'] = data_df['title'] + ' ' + data_df['text']

# %% Preprocess real news.  
real_news = data_df[data_df['label'] == 1]['content']
fake_news = data_df[data_df['label'] == 0]['content']

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

real_news = real_news.apply(preprocess)
fake_news = fake_news.apply(preprocess)

# %% Extract key phrases. 
real_news_phrases = TfidfVectorizer(
    ngram_range=(3, 5), stop_words="english", max_features=20
)
real_news_phrases.fit_transform(real_news)

real_news_phrases = real_news_phrases.get_feature_names_out()

fake_news_phrases = TfidfVectorizer(
    ngram_range=(3, 5), stop_words="english", max_features=20
)
fake_news_phrases.fit_transform(fake_news)

# Remove terms from fake news. 
fake_news_phrases = fake_news_phrases.get_feature_names_out()

# %% Save 
import dill as pickle

with open("./data/real_news_phrases.pkl", "wb") as file:
    pickle.dump(real_news_phrases, file)

with open("./data/fake_news_phrases.pkl", "wb") as file:
    pickle.dump(fake_news_phrases, file)