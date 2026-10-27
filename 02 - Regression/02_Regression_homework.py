# %% Import modules and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "car_fuel_efficiency_2026.csv"
df_CarData = pd.read_csv(data_path)

df_CarData.head()

# %% Prepare data --> copy original dataframe and export feature matrix
"""
Use only the following columns:

'engine_displacement',
'horsepower',
'vehicle_weight',
'model_year',
'fuel_efficiency_mpg'

Look at the fuel_efficiency_mpg variable. Does it have a long tail?

"""
# First, slice df using the above mentioned columns only
base = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df_base = df_CarData[base]
print(df_base.head())


# EDA --> Look at the fuel_efficiency_mpg variable
sns.histplot(df_base['fuel_efficiency_mpg'])
plt.show()
# Does not see any tail. no need for log1p transform?

# %% Q1 - There's one column with missing values. What is it?

print(df_base.isnull().sum())
# Column = horsepower

# %% Q2 - What's the median (50% percentile) for variable 'horsepower'?

horsepower_med = df_base['horsepower'].median()
print(f"Median horsepower: {horsepower_med}") # 254


# %% Prepare and split the dataset
# First,split data into train, val and test AND split into X and target variable y

# Creating sized for each selection
n = len(df_base)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - n_val - n_test

print(f"Total data size: {n},\nValidation size: {n_val},\nTest size: {n_test},\nTraining size: {n_train}")

# Slicing after random shuffle
np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df_base.iloc[idx[:n_train]]
df_val = df_base.iloc[idx[n_train:n_train+n_val]]
df_test = df_base.iloc[idx[n_train+n_val:]]

# Reset index to fit each selection
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Extract the corresponding y values for each selection and afterwards remove from selection data to create X
y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values
y_test = df_test['fuel_efficiency_mpg'].values

df_train = df_train.drop('fuel_efficiency_mpg', axis=1)
df_val = df_val.drop('fuel_efficiency_mpg', axis=1)
df_test = df_test.drop('fuel_efficiency_mpg', axis=1)
# %% Q3 - build model and test
"""
We need to deal with missing values for the column from Q1.
We have two options: fill it with 0 or with the mean of this variable.
Try both options. For each, train a linear regression model without regularization using the code from the lessons.
For computing the mean, use the training only!
Use the validation dataset to evaluate the models and compare the RMSE of each option.
Round the RMSE scores to 3 decimal digits using round(score, 3). This keeps the imputation difference visible in this release.
Which option gives better RMSE?
"""

def train_lin_reg(X, y):
    ones = np.ones(X.shape[0]) # Add bias weight = 1
    X = np.column_stack([ones, X])
    
    # Gram matrix and linear system:
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    w0 = w_full[0]
    w = w_full[1:]
    return w0, w

def prepare_X(df):
    df_num = df.fillna(0)
    X = df_num.values
    return X

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return round(np.sqrt(mse), 4)


    
train_mean = round(df_train['horsepower'].mean(), 2)

df_train_mean = df_train.copy()
df_train_mean['horsepower'] = df_train_mean['horsepower'].fillna(train_mean)

df_val_mean = df_val.copy()
df_val_mean['horsepower'] = df_val_mean['horsepower'].fillna(train_mean)

df_test_mean = df_test.copy()
df_test_mean['horsepower'] = df_test_mean['horsepower'].fillna(train_mean)

# Calculate the two different models --> M1 = .fillna(0) and M2 = .fillna(mean)
### M1 ###
X_train_0 = prepare_X(df_train)
w0, w = train_lin_reg(X_train_0, y_train)

X_val_0 = prepare_X(df_val)
y_pred_0 = w0 + X_val_0.dot(w)
y_pred_0_rmse = rmse(y_val, y_pred_0)
print(f"RMSE for validation with .fillna(0): {y_pred_0_rmse}") # RMSE = 2.205

### M2 ###
X_train_mean = prepare_X(df_train_mean)
w0, w = train_lin_reg(X_train_mean, y_train)

X_val_mean = prepare_X(df_val_mean)
y_pred_mean = w0 + X_val_mean.dot(w)
y_pred_mean_rmse = rmse(y_val, y_pred_mean)
print(f"RMSE for validation with .fillna(mean): {y_pred_mean_rmse}") # RMSE = 2.202

# %% Q4 - regularization

"""
Now let's train a regularized linear regression.
For this question, fill the NAs with 0.
Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
Use RMSE to evaluate the model on the validation dataset.
Round the RMSE scores to 4 decimal digits. This keeps the small but real regularization differences visible instead of turning several choices into a tie.
Which r gives the best RMSE?
"""
def train_lin_regression_reg(X, y, r):
    ones = np.ones(X.shape[0]) # Add bias weight = 1
    X = np.column_stack([ones, X])
    
    # Gram matrix and linear system:
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    w0 = w_full[0]
    w = w_full[1:]
    return w0, w

# Tuning model based on different regularization values

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    X_train = prepare_X(df_train)
    
    w0, w = train_lin_regression_reg(X_train, y_train, r=r)
    
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    
    print(f"r: {r} \nw0: {w0} \nscore: {score}")
    print('-------------------------')

# r = 0 is the best with rmse = 2.2053

# %% Q5 - random selection seed impact
# Make a function to use for the different seed values

def split_data(df, seed_num):
    n = len(df)
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - n_val - n_test
    
    np.random.seed(seed_num)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    # Reset index to fit each selection
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Extract the corresponding y values for each selection and afterwards remove from selection data to create X
    y_train = df_train['fuel_efficiency_mpg'].values
    y_val = df_val['fuel_efficiency_mpg'].values
    y_test = df_test['fuel_efficiency_mpg'].values

    df_train = df_train.drop('fuel_efficiency_mpg', axis=1)
    df_val = df_val.drop('fuel_efficiency_mpg', axis=1)
    df_test = df_test.drop('fuel_efficiency_mpg', axis=1)
    
    return df_train, df_val, df_test, y_train, y_val, y_test


rmse_scores = []
for val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_base, val)
    X_train = prepare_X(df_train)
    
    w0, w = train_lin_reg(X_train, y_train)
    
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    rmse_scores.append(rmse(y_val, y_pred))

print(f"Standard deviation of different seed values: {round(np.std(rmse_scores), 3)}") # 0.029

# %% Q6 - test model on test data after combining training and validation data
"""
Split the dataset like previously, use seed 9.
Combine train and validation datasets.
Fill the missing values with 0 and train a model with r=0.001.
What's the RMSE on the test dataset?
"""

