import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

# training_features = pd.read_csv('c:\\Users\\MV\\Desktop\\ml-pipeline-demo\\data\\processed\\train_processed.csv')
# test_features = pd.read_csv('c:\\Users\\MV\\Desktop\\ml-pipeline-demo\\data\\processed\\test_processed.csv')

training_features = pd.read_csv('./data/interim/train_processed.csv')
test_features = pd.read_csv('./data/interim/test_processed.csv')

# training_features = training_features.fillna('',inplace=True)
# test_features = test_features.fillna('', inplace=True)

x_train = training_features['content'].values
y_train = training_features['sentiment'].values

x_test = test_features['content'].values
y_test = test_features['sentiment'].values


vectorizer = TfidfVectorizer(max_features=max_features)

x_train_bow = vectorizer.fit_transform(x_train)
x_test_bow = vectorizer.fit_transform(x_test)
print(type(x_train_bow))

train_df = pd.DataFrame(x_train_bow.toarray())
train_df['labels'] = y_train

test_df = pd.DataFrame(x_test_bow.toarray())
test_df['labels'] = y_test




data_path = os.path.join("data","processed")
os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train_tfidf.csv"))
test_df.to_csv(os.path.join(data_path,"test_tfidf.csv"))

