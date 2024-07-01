import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import os

# fetch the data from raw folder
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Join tokens back into string
    preprocessed_tweet = ' '.join(lemmatized_tokens)
    
    return preprocessed_tweet

def normalize(df):
    df['processed_tweet'] = df['content'].apply(preprocess_tweet)
    return df

# Sample training dataframe
# data = {
#     'tweet': [
#         "Just received my new iPhone! #excited https://example.com",
#         "Feeling happy after a long day at work. #relaxed",
#         "Can't believe it's already Friday! #weekendvibes"
#     ],
#     'label': [1, 0, 1]  # Example labels
# }
# df = pd.DataFrame(data)

# Apply preprocessing and normalization
# df = normalize(df)

# Show the processed dataframe
# print(df)

train_processed_data = normalize(train_data)
test_processed_data = normalize(test_data)




data_path = os.path.join("data","interim")
os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))