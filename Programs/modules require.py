# ==============================================================================
# TERMINAL INSTALLATION COMMAND:
# Before running this script, install the libraries via your terminal:
# pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. DATA MANIPULATION & MATHEMATICS IMPORTS
# ------------------------------------------------------------------------------
import numpy as np  # Industry standard alias 'np' for numerical computing
import pandas as pd  # Industry standard alias 'pd' for dataframes

# --- Operations & Syntax ---
# NumPy: Create a 1D array (vector) and calculate mathematical metrics
matrix_data = np.array([[1, 2, 3], [4, 5, 6]])
matrix_mean = np.mean(matrix_data)  # Operation: Computes global average (3.5)

# Pandas: Create a structured DataFrame from scratch (simulating a CSV file)
raw_data = {"Feature_1": [10, 20, 30, 40], "Feature_2": [5, 15, 25, 35], "Target": [0, 1, 0, 1]}
df = pd.DataFrame(raw_data)
first_rows = df.head(2)  # Operation: Extracts the first 2 rows of the data


# ------------------------------------------------------------------------------
# 2. DATA VISUALIZATION IMPORTS
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt  # Standard alias 'plt' for basic plotting
import seaborn as sns  # Standard alias 'sns' for advanced statistical plots

# --- Operations & Syntax ---
# Matplotlib: Set up a basic canvas layout
plt.figure(figsize=(5, 5))

# Seaborn: Generate a correlation heatmap using the Pandas DataFrame
# Operation: Calculates correlations and plots them visually
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Data Correlation Matrix")
# plt.show() # Operation: Renders the plot window (uncomment to display)


# ------------------------------------------------------------------------------
# 3. TRADITIONAL MACHINE LEARNING IMPORTS (Scikit-Learn)
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split  # Module to split datasets
from sklearn.linear_model import LogisticRegression  # A classification algorithm
from sklearn.metrics import accuracy_score  # Metric to evaluate model performance

# --- Operations & Syntax ---
# Separate the DataFrame into features (X) and target/label (y)
X = df[["Feature_1", "Feature_2"]]
y = df["Target"]

# Operation: Split data into 75% training data and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Operation: Initialize and train the ML model
ml_model = LogisticRegression()
ml_model.fit(X_train, y_train)  # Training step

# Operation: Test the model by making predictions on unseen test data
predictions = ml_model.predict(X_test)
score = accuracy_score(y_test, predictions)  # Calculates percentage of correct guesses


# ------------------------------------------------------------------------------
# 4. DEEP LEARNING IMPORTS (Option A: TensorFlow & Keras)
# ------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras

# --- Operations & Syntax ---
# Operation: Constructing a basic Deep Learning Neural Network architecture
tf_model = keras.Sequential([
    # Input layer expecting 2 features, routing into a dense layer of 8 neurons
    keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    # Output layer with 1 neuron utilizing 'sigmoid' output for binary classification
    keras.layers.Dense(1, activation='sigmoid')
])

# Operation: Configuring the training settings (optimizer, loss math, and tracking metric)
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ------------------------------------------------------------------------------
# 5. DEEP LEARNING IMPORTS (Option B: PyTorch)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn

# --- Operations & Syntax ---
# Operation: Device Agnostic Code setup (checks if an NVIDIA graphics card is usable)
compute_device = "cuda" if torch.cuda.is_available() else "cpu"

# Operation: Create a data tensor directly inside the allocated hardware memory (CPU or GPU)
tensor_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(compute_device)


# ------------------------------------------------------------------------------
# CONSOLE confirmation check
# ------------------------------------------------------------------------------
print("--- ALL IMPORTS AND TEST OPERATIONS EXECUTED SUCCESSFULY ---")
print(f"PyTorch Device Status: Using {compute_device.upper()} for calculations.")
print(f"Pandas Data Sample Processed:\n{first_rows}")
