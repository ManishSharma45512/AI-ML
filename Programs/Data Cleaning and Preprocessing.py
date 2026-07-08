
#
#          CSV:    Sample.CSV -------------------->Used in This
#
#




import pandas as pd
import numpy as np

# Create a dummy dataset with missing values, outliers, and categorical data
data = {
    'Age': [25, 30, np.nan, 35, 22, 28, 150, 32, 29, np.nan],  # 150 is an outlier
    'Salary': [50000, 60000, 55000, np.nan, 45000, 1200000, 62000, 58000, np.nan, 51000], # 1200000 is an outlier
    'Department': ['HR', 'IT', 'IT', 'Marketing', 'HR', 'Marketing', 'IT', 'HR', 'IT', 'Marketing']
}

df_mock = pd.DataFrame(data)
df_mock.to_csv('sample_data.csv', index=False)
print("Sample data 'sample_data.csv' created successfully!")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Installing all the module

!pip install pandas numpy scikit-learn missingno


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Step-by-Step Pipeline Execution

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno

# 1. Load the dataset
df = pd.read_csv('sample_data.csv')
print("--- Original Data ---")
print(df.head(10))

# 2. Visualize missing data
print("\n--- Visualizing Missing Data ---")
msno.matrix(df)

# 3. Handle missing values 
# We isolate numeric columns to safely apply the mean fill
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical missing values if any exist (using mode)
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n--- After Filling Missing Values ---")
print(df)

# 4. Handle outliers using Z-score (Only on numeric columns)
print("\n--- Removing Outliers ---")
z_scores = np.abs(stats.zscore(df[numeric_cols]))
# Keep rows where all numeric features have a Z-score less than 2 (adjusted to 2 for small mock dataset)
df_no_outliers = df[(z_scores < 2).all(axis=1)].reset_index(drop=True)
print(df_no_outliers)

# 5. Scale and normalize numerical data
print("\n--- Scaling Numerical Data ---")
scaler = StandardScaler()
df_scaled = df_no_outliers.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_no_outliers[numeric_cols])
print(df_scaled)

# 6. Encode categorical variables
print("\n--- One-Hot Encoding ---")
df_encoded = pd.get_dummies(df_scaled, columns=['Department'], dtype=int)
print(df_encoded)

# 7. Save the cleaned and preprocessed data
df_encoded.to_csv('cleaned_preprocessed_data.csv', index=False)
print('\nData cleaning complete. File saved as "cleaned_preprocessed_data.csv"')



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Automated & Reusable Workflow Functions


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def remove_outliers(df, threshold=2):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    return df[(z_scores < threshold).all(axis=1)].reset_index(drop=True)

def scale_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns, dtype=int)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)

# Execution using the reusable functions:
raw_df = load_data('sample_data.csv')
clean_df = handle_missing_values(raw_df)
clean_df = remove_outliers(clean_df, threshold=2)
clean_df = scale_data(clean_df)
final_df = encode_categorical(clean_df, ['Department'])
save_data(final_df, 'automated_output.csv')

print("Automated pipeline finished successfully! Saved output to 'automated_output.csv'")
print(final_df)


#------------------------XXXXXXXXXXXXXXX--------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX------------------------------------------XXXXXXXXXXXXXXXX----------------------------------------
