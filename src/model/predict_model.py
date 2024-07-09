import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,classification_report
import json
from dvclive import Live
import yaml

test_df = pd.read_csv('./data/processed/test_bow.csv')

x_test = test_df.iloc[:,0:-1].values
y_test = test_df.iloc[:,-1].values

clf = pickle.load(open('models/model.pkl','rb'))

y_pred = clf.predict(x_test)
y_pred_prob = clf.predict_proba(x_test)[:,1]

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_pred)


with open('params.yaml','r') as file:
    params = yaml.safe_load(file)


with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy',accuracy)
    live.log_metric('precision',precision)
    live.log_metric('recall',recall)
    live.log_metric('auc',auc)

    for param,value in params.items():
        for key,val in value.items():
            live.log_param(f'{param}_{key}',val)


metrics_dict = {
    'accuracy':accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}

import json
with open('reports/metrics.json','w') as file:
    json.dump(metrics_dict,file,indent=4)