import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %% Dataset

data_path = "course_lead_scoring_2026.csv"
df = pd.read_csv(data_path)

# %% Data preparation

df.columns = df.columns.str.lower().str.replace(' ', '_')
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
for c in cat_cols:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Check how many missing valies there is 
print(df.isnull().sum())

# If missing value is object/categorical, replace with 'NA' and if numerical replace with 0
# cat_cols already stores all columnsn that are have string values
for c in df.columns:
    if c in cat_cols:
        df[c] = df[c].fillna('NA')
    else:
        df[c] = df[c].fillna(0)

print(df.isnull().sum())

# %% Q1 - What is the most frequent observation (mode) for the column industry?    

print(df['industry'].value_counts()) # technology with 1173

# %% Q2 - Question 2 - Create the correlation matrix for the numerical features of your dataset

# Extract the columns that are numerical values
num_cols = df.select_dtypes(include='number').columns.tolist()

# Creates the correlation matrix in full, with the diagonal being 1's
corr_matrix = df[num_cols].corr().abs()
# Doing it with corrwith requires two lines:
print(f"Lead_score correlation: \n{df[num_cols].corrwith(df['lead_score']).abs()}") # Gives the correlation between lead_score and interaction_count and number_of_courses_viewed

print(f"Interaction:count correletiaon: \n{df[num_cols].corrwith(df['interaction_count']).abs()}")  # Gives the correlation between interaction_count and lead_score and number_of_courses_viewed

#Biggest correlation is interaction_count and lead_score

# %% Q3 - Mutual informaiton
"""
Calculate the mutual information score between converted and other categorical variables in the dataset. Use the training set only.
Round the scores to 2 decimals using round(score, 2).
"""

# First, split data
df_train_full, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df_train, df_val = train_test_split(df_train_full, test_size = 0.25, random_state = 42)

# Remember to reset the index
df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Extract the target value from all datasets
y_train_full = df_train_full['converted']
y_test = df_test['converted']
y_train = df_train['converted']
y_val = df_val['converted']

# Delete the target from training datasets
del df_train_full['converted']
del df_test['converted']
del df_train['converted']
del df_val['converted']

# Calculating the MI between the target variable "converted" and the other categorial features
def mutual_info_converted_score(series):
    return round(mutual_info_score(series, y_train), 3) # to use the converted, we need to specify the y_train because we have deleted above. 
mi = df_train[cat_cols].apply(mutual_info_converted_score)
mi.sort_values(ascending=False) # lead_source = 0.034


# %% Q4 - Train LR
"""
Now let's train a logistic regression.
Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
Fit the model on the training dataset.
To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
"""

# Implement One-hot-encoder to the training set
num_cols_train = num_cols.copy()
num_cols_train.remove('converted')
# Define the dict-to-OHE using the built-in function
dv = DictVectorizer(sparse = False)
train_dict = df_train[cat_cols + num_cols_train].to_dict(orient='records')

X_train = dv.fit_transform(train_dict)
print(dv.get_feature_names_out())
X_train.shape


# Doing the same for the validation dataset
val_dict = df_val[cat_cols + num_cols_train].to_dict(orient = 'records')
X_val = dv.transform(val_dict)

# Defining the model to use for the training
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
# Now calculate the bias and weights 
model.fit(X_train, y_train)
w0 = model.intercept_[0]
w = model.coef_[0].round(3)

# Predict using the trained model
y_pred = model.predict(X_val) # Only extract the probabilty for class = 1

model_acc = round((y_pred == y_val).mean(), 2)
print(f" Model accuracy on validation: {model_acc}")

# %% Q5 - finding the feature that has the least impact on accuracy
# First, define a function to use for the above code

def train_evaluate(features):
    dv = DictVectorizer(sparse = False)
    train_dict = df_train[features].to_dict(orient='records')
    val_dict = df_val[features].to_dict(orient='records')
    
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    acc = (y_pred == y_val).mean()
    
    return acc

features = cat_cols + num_cols_train
original_acc = train_evaluate(features)

results = {}
for i, feature_missing in enumerate(features):
    features_least = features.copy()
    features_least.pop(i)

    feature_acc = train_evaluate(features_least)
    results[feature_missing] = original_acc - feature_acc

pd.Series(results).sort_values()
          
# %% Q6 - train regularised LogReg and test different C-values

def LogReg_reg_train_evaluate(features, C):
    dv = DictVectorizer(sparse = False)
    train_dict = df_train[features].to_dict(orient='records')
    val_dict = df_val[features].to_dict(orient='records')
    
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    acc = (y_pred == y_val).mean()
    
    return acc

C_list = [0.000001, 0.00001, 0.0001, 0.001]

results = {}
for c in C_list:
    model_acc = LogReg_reg_train_evaluate(features, c)
    
    results[c] = round(model_acc, 3)

print(pd.Series(results))

pd.Series(results).idxmax() # 0.001



    
    
    