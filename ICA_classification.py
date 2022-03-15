import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

files = ['Clc6.xls', 'Clc8.xls', 'Clc15.xls']
KMI_cols = ['E%c_KMI_8_13', 'E%c_KMI_15_30']
VMI_cols = ['E%c_VMI_8_13', 'E%c_VMI_15_30']
conditions = ['C', 'O']
repeat_num = 200



classifiers = [SVC(kernel='linear'), LogisticRegression(), 
               KNeighborsClassifier(), RandomForestClassifier(),]
results = {}
for condition in conditions:
    for file in files:
        print(file)
        ica = pd.read_excel(file)
        vals = ~np.isnan(ica['subj'].values)
        ica = ica[vals]
        KMI_mask = [col%condition for col in KMI_cols]
        VMI_mask = [col%condition for col in VMI_cols]
        X = np.concatenate([ica[KMI_mask].values,
                            ica[VMI_mask].values], axis=0)
        y = np.concatenate([np.zeros(ica.shape[0]), np.ones(ica.shape[0])], axis=0)
        for classifier in classifiers:
            print(type(classifier).__name__)
            pipe = Pipeline([['scaler', StandardScaler()], ['classifier',classifier]])
            res = [cross_val_score(pipe, X, y, scoring='accuracy',
                            cv = StratifiedKFold(5, shuffle=True)) 
                 for i in range(repeat_num)]
            res = np.array(res).mean(axis = -1)
            results[f'{condition}_{file}_{type(classifier).__name__}'] = res

import pickle
with open('classification_results_new.pkl', 'wb') as file:
    pickle.dump(results, file)
    
for key in results.keys():
    print(key, results[key].mean(), results[key].std())
    
for file in files:
    fig, ax = plt.subplots()
col_names = []
cols = []
for key in results.keys():
    col_names.append(key)
    cols.append(results[key])
to_excel = pd.DataFrame(np.stack(cols, axis = -1), columns = col_names)
to_excel.to_excel('classification_comparison_new_new_02.12.2021.xlsx', index=False)