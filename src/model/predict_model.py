import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,classification_report
import json

test_df = pd.read_csv('./data/processed/test_tfidf.csv')

x_test = test_df.iloc[:,0:-1].values
y_test = test_df.iloc[:,-1].values

clf = pickle.load(open('models/model.pkl','rb'))

y_pred = clf.predict(x_test)
y_pred_prob = clf.predict_proba(x_test)[:,1]

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_pred)


metrics_dict = {
    'accuracy':accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}

import json
with open('reports/metrics.json','w') as file:
    json.dump(metrics_dict,file,indent=4)