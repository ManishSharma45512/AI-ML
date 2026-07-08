def load_data(df):
    return df

def handle_missing_values(df):
    return df.fillna(df.mean())  # For numeric data, fill missing values with the mean

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]  # Remove rows with any outliers

def scale_data(df):
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)

#These functions encapsulate the core preprocessing tasks, making them reusable across different datasets. They will be applied to our dummy data.


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load the data
df_preprocessed = load_data(df_dummy)

# Handle missing values
df_preprocessed = handle_missing_values(df_preprocessed)

# Remove outliers
df_preprocessed = remove_outliers(df_preprocessed)

# Scale the data
df_preprocessed = scale_data(df_preprocessed)

# Encode categorical variables
df_preprocessed = encode_categorical(df_preprocessed, ['Category'])

# Display the preprocessed data
print(df_preprocessed.head())



'''This code applies the preprocessing steps to the dummy data. It handles missing values by filling them with the mean,
   removes outliers using the Z-score method, scales the numeric data, and encodes the categorical variables using one-hot encoding.'''


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
