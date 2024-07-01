
import pandas as pd
import numpy as np
import pickle
import yaml

params = yaml.safe_load(open('params.yaml','r'))['model_building']
from sklearn.ensemble import GradientBoostingClassifier

train_df = pd.read_csv('./data/processed/train_bow.csv')

x_train = train_df.iloc[:,0:-1].values
y_train = train_df.iloc[:,-1].values

clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
clf.fit(x_train,y_train)

pickle.dump(clf, open('models/model.pkl','wb'))