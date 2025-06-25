# train.py
# This script trains the model and saves the entire pipeline.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib # Used to save the model

# --- 1. Load the Data ---
try:
    df = pd.read_csv('bank.csv', sep=';')    
except FileNotFoundError:    
    exit()

# --- 2. Data Preprocessing ---
# Define features (X) and target (y)
X = df.drop('y', axis=1)
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create the preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. Build the Model ---
# We use a RandomForestClassifier.
# The pipeline includes preprocessing and the classifier.
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

# --- 4. Train the Model ---
# Train the model on the entire dataset.
model_pipeline.fit(X, y)


# --- 5. Save the Model ---
# Save the entire pipeline to a file.
model_filename = 'model.joblib'
joblib.dump(model_pipeline, model_filename)


