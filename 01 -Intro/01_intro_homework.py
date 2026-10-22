import pandas as pd
import numpy as np
import seaborn as sns

# %% Q1 - What version of Pandas did you install?
print(pd.__version__)

# %% Q2 - How many records are in the dataset?

## Read .csv data file
data_path = "car_fuel_efficiency_2026.csv"
data = pd.read_csv(data_path)

num_records = data.shape[0]
print(f"Number of records: {num_records}")

# %% Q3 - How many fuel types are presented in the dataset?
# number of unique values in the column "fuel_type"

# Unique values:
fuel_type_unique = data['fuel_type'].unique()
# number of unique values
fuel_type_nunique = data['fuel_type'].nunique()
# Count of how many for each unique value
fuel_type_count = data['fuel_type'].value_counts()

print(f"number of fuel types in the dataset: {fuel_type_nunique}")

# %% Q4 - How many columns in the dataset have missing values?
col_NaN = data.isna().any().sum()
print(f"Number of columns wiht missing values: {col_NaN}")

# %% Q5 - What's the maximum fuel efficiency of cars from Asia?
Asian_cars = data[data['origin'] == 'Asia']
Asian_cars_max_FE = Asian_cars['fuel_efficiency_mpg'].max()
print(f"Maxium fuel efficientcy of cars from Asia: {Asian_cars_max_FE}")

# %% Q6 - Median value of horsepower
"""
Find the median value of the horsepower column in the dataset.
Next, calculate the most frequent value of the same horsepower column.
Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
Now, calculate the median value of horsepower once again.
"""
# Find median
horsepower_median = data['horsepower'].median()
print(f"median horsepower: {horsepower_median}")
# Find most frequent value
horsepower_freq_valueCounts = data['horsepower'].value_counts().idxmax()
print(f"most frequent horsepower value: {horsepower_freq_valueCounts}")
# Use fillna with most frequent value
data['horsepower'].fillna(horsepower_freq_valueCounts)
# calculate new median
horsepower_median_Post_Fillna = data['horsepower'].median()
print(f"median horsepower post .fillna: {horsepower_median_Post_Fillna}")

# %% Q7 - Sum of weights
"""
Select all the cars from Asia
Select only columns vehicle_weight and model_year
Select the first 7 values
Get the underlying NumPy array. Let's call it X.
Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
Invert XTX.
Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
What's the sum of all the elements of the result?

"""
Asian_cars_2Col = Asian_cars[['vehicle_weight', 'model_year']]
Asian_cars_2Col_7rows = Asian_cars_2Col.iloc[:7]

X = Asian_cars_2Col_7rows.to_numpy()

XTX = X.T.dot(X)

invertXTX = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = invertXTX @ X.T @ y

w.sum()