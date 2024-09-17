import pandas as pd
import numpy as np
import uuid
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers

# Load the synthetic data
candidates_df = pd.read_csv('synthetic_data/candidates.csv')
jobs_df = pd.read_csv('synthetic_data/jobs.csv')
candidate_skills_df = pd.read_csv('synthetic_data/candidate_skills.csv')
job_skills_df = pd.read_csv('synthetic_data/job_skills.csv')

# Step 1: Prepare the Data
# Merge candidate and job data
data = []
for _, candidate in candidates_df.iterrows():
  for _, job in jobs_df.iterrows():
      # Calculate match percentage based on skills
      candidate_skills = candidate_skills_df[candidate_skills_df['candidate_id'] == candidate['candidate_id']]
      job_skills = job_skills_df[job_skills_df['job_id'] == job['job_id']]
      
      # Calculate match based on common skills
      common_skills = set(candidate_skills['skill_id']).intersection(set(job_skills['skill_id']))
      match_percentage = len(common_skills) / len(job_skills) * 100 if len(job_skills) > 0 else 0
      
      data.append({
          'candidate_id': candidate['candidate_id'],
          'job_id': job['job_id'],
          'match_percentage': match_percentage,
          'desired_salary': candidate['desired_salary'],
          'job_type': job['job_type'],
          'location': candidate['location']
      })

# Create a DataFrame from the data
match_df = pd.DataFrame(data)

# Step 2: Build the Neural Network
# Define features and target
X = match_df[['desired_salary', 'job_type', 'location']]
y = match_df['match_percentage']

# Preprocessing
categorical_features = ['job_type', 'location']
numerical_features = ['desired_salary']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
  transformers=[
      ('num', StandardScaler(), numerical_features),
      ('cat', OneHotEncoder(), categorical_features)
  ])

# Create a pipeline that first transforms the data and then fits the model
model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('nn', keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=(None,)),
      layers.Dense(32, activation='relu'),
      layers.Dense(1, activation='sigmoid')  # Output layer for match percentage
  ]))
])

# Step 3: Train the Model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compile the model
model.named_steps['nn'].compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Step 4: Evaluate the Model
loss, mae = model.named_steps['nn'].evaluate(X_test, y_test)
print(f"Mean Absolute Error on Test Set: {mae}")

# Step 5: Make Predictions
# Example: Predict match percentage for a new candidate-job pair
new_data = pd.DataFrame({
  'desired_salary': [70000],
  'job_type': ['Full-time'],
  'location': ['New York']
})

predicted_match_percentage = model.predict(new_data)
print(f"Predicted Match Percentage: {predicted_match_percentage[0][0] * 100:.2f}%")