import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# %% Dataset

data_path = "course_lead_scoring_2026.csv"
df = pd.read_csv(data_path)

# %% Data preparation
"""
Check if the missing values are presented in the features.
If there are missing values:
For categorical features, replace them with 'NA'
For numerical features, replace with with 0.0
Split the data into 3 parts
"""

# Columns
df.columns = df.columns.str.lower().str.replace(' ', '_')
# Object/string values and Missing values
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = df.select_dtypes(include=['number']).columns.tolist()

print(f"Before missing values process: \n{df.isnull().sum()}")
for c in cat_cols:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    df[c] = df[c].fillna('NA')

for c in num_cols:
    df[c] = df[c].fillna(0.0)

print(f"After missing values process: \n{df.isnull().sum()}")

# Split data into 3 parts
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Create y feature vectors
#y_full_train = df_full_train['converted']
y_train = df_train['converted']
y_val = df_val['converted']
y_test = df_test['converted']

#del df_full_train['converted']
del df_test['converted']
del df_train['converted']
del df_val['converted']


# %% Q1 - ROC AUC feature importance
# instead of using a models predictions (that is based on several features), we now try and see how a 
# single feature "correlates" to the target variable "converted".
from sklearn.metrics import roc_auc_score
# Get all numerical columns from the training set
num_cols_training = df_train.select_dtypes(include=['number']).columns.tolist()

for c in num_cols_training:
    auc = roc_auc_score(y_train, df_train[c]) 
    print(f"{c}: {auc:.3f}") # lead_score: 0.788

# %% Q2 - train the model (logistic regression)

# Define a function to use for the OHE
def OHE_dv(df_train, df_val):
    dv = DictVectorizer(sparse = False)
    # Convert df to dict
    train_dict = df_train.to_dict(orient='records')
    val_dict = df_val.to_dict(orient='records')
    # Create the OHE version of the df
    X_train = dv.fit_transform(train_dict)
    
    # Use the same conversion on the validaiton df as the training to ensure comparability
    X_val = dv.transform(val_dict)
    
    return X_train, X_val

X_train, X_val = OHE_dv(df_train, df_val)

# Define the model from the assignment
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    

# train the model
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1] # returns 2 columns, first column is P(not converted) and second column is P(converted)

# Get the AUC score

auc = roc_auc_score(y_val, y_pred)
print(f"AUC: {auc:.3f}") # AUC: 0.732

# %% Q3 - Precission and recall

thres = np.linspace(0,1,101)

# Empyt list for storing tuples inside
scores = []
for t in thres:
    prediction = y_pred >= t
    
    TP = ((y_val == 1) & (prediction == 1)).sum()
    FP = ((y_val == 0) & (prediction == 1)).sum()
    TN = ((y_val == 0) & (prediction == 0)).sum()
    FN = ((y_val == 1) & (prediction == 0)).sum()
    
    if TP + FP == 0:
        P = np.nan
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
    
    scores.append((round(t,2), TP, FP, FN, TN, round(P,2), round(R,2)))

columns = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall'] 
df_scores = pd.DataFrame(scores, columns=columns)
# Adding the difference column
df_scores['diff'] = (df_scores['precision'] - df_scores['recall']).abs()

# Plot the different P and recall values against the tresholds
plt.plot(df_scores['threshold'], df_scores['precision'])
plt.plot(df_scores['threshold'], df_scores['recall'])
plt.xlabel('threshold')
plt.legend(['precision', 'recall'])
plt.show()

# find the threshold with the smallest difference between precision and call
min_diff_idx = df_scores['diff'].idxmin()
min_diff_t = df_scores.loc[min_diff_idx]
print(f"threshold value for the lowest diff: {round(min_diff_t['threshold'], 2)}") # t = 0.63

# %% Q4 - F1 score
# Calculate F1 column for all thresholds

df_scores['F1'] = 2 * (df_scores['precision'] * df_scores['recall'])/((df_scores['precision'] + df_scores['recall']))

max_f1_idx = df_scores['F1'].idxmax()
max_f1_t = df_scores.loc[max_f1_idx]
print(f"threshold with max F1: {round(max_f1_t['threshold'], 2)}") # t = 0.41

# %% Q5 - 5-fold CV

# Define function to use for dv + training model

def model_train(df_train, y_train, C):
    dv = DictVectorizer(sparse=False)
    
    dicts = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    # convert e.g. test df into dict for dv usage
    dicts = df.to_dict(orient='records')
    # use the dv already created
    X = dv.transform(dicts)
    
    # use the trained model
    y_pred = model.predict_proba(X)[:,1] # only extract the "1" predictions
    
    return y_pred

scores = [] # empty list to store results
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train['converted']
    y_val = df_val['converted']
    
    dv, model = model_train(df_train.drop(columns='converted'), y_train, C=1.0)
    y_pred = predict(df_val.drop(columns='converted'), dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print(f"across 5-KFOld CV:  Mean score: {round(np.mean(scores), 3)},  std score: {round(np.std(scores), 3)}")
# std = 0.007

# %% Q7 - Hyperparameter tuning

for c in [0.000001, 0.001, 1]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    
    scores_c = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        
        y_train = df_train['converted']
        y_val = df_val['converted']
        
        dv, model = model_train(df_train.drop(columns='converted'), y_train, C=c)
        y_pred = predict(df_val.drop(columns='converted'), dv, model)
        
        auc = roc_auc_score(y_val, y_pred)
        scores_c.append(auc)
    
    print('C=%s %.3f +- %.3f' % (c, np.mean(scores_c), np.std(scores_c)))
    
    # Best AUC ROC is c = 0.001 with C=0.001 0.749 +- 0.010
