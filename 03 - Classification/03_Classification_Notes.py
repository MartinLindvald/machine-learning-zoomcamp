import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

df.head()

# %% Prepare Data
# Standarize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Standarize string values
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
for c in cat_cols:
    df[c] = df[c].str.lower().str.replace(' ', '_')

print(df.dtypes)
# totalcharges are object and not int. we need to convert this.
tc = pd.to_numeric(df['totalcharges'], errors='coerce')
# Look if Nan
tc.isnull().sum() # 11 entries with NaN
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['totalcharges'] = df['totalcharges'].fillna(0)

df['totalcharges'].isnull().sum() # now 0 NaN entries

# Convert the y array (churn column) that is strings/object to numbers as this is needed for the y-array
# First ensure that only yes and no exist in the churn column:
print(df['churn'].unique()) # only no and yes and no YES or Yes or No
df['churn'] = (df['churn'] == 'yes').astype(int)


# %% Split into training, validation and test datasets

df_train_full, df_test = train_test_split(df, train_size=0.8, random_state = 1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)
# Now, reset the index for each dataset
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train_full = df_train.reset_index(drop=True)


# Extract the target y array from the datasets
y_train = df_train['churn'].values
y_val = df_val['churn'].values
y_test = df_test['churn'].values

# Delete the target variable from the feature matrices
del df_train['churn']
del df_val['churn']
del df_test['churn']

# %% EDA (Explorative Data analysis)
# Good practive to start by checking for Nan or missing values
print(df_train_full.isnull().sum())

# look at the target variable:
print(df_train_full['churn'].value_counts(normalize=True))


# Categorical columns:
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]
# becuase they are categorical, we need to see how many categories/unique values per column before transforming into one-hot encoding
df_train_full[categorical].nunique()

# %% Risk Ratio EDA
# Analyse which features might impact the churn score the most
global_churn = df_train_full['churn'].mean()

for c in categorical:
    print(c)
    df_group = df_train_full.groupby(c)['churn'].agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    print(df_group)
    print()
    print()

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_train_full['churn'])

mi = df_train_full[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)

# %% Correlation on numerical values (similar to risk-ratio from categorical values) to see which featuer impacts the target feature y
   

    