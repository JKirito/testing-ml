import pandas as pd
import numpy as np
import uuid
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
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

# Encode candidate and job skills
# Map skills to each candidate
candidate_skills_map = candidate_skills_df.groupby('candidate_id')['skill_id'].apply(list).reset_index()
# Map skills to each job
job_skills_map = job_skills_df.groupby('job_id')['skill_id'].apply(list).reset_index()

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# Fit on all unique skills
all_skills = pd.concat([candidate_skills_df['skill_id'], job_skills_df['skill_id']]).unique()
mlb.fit([all_skills])

# Transform candidate skills
candidate_skills_map['skills_encoded'] = mlb.transform(candidate_skills_map['skill_id']).tolist()
# Transform job skills
job_skills_map['skills_encoded'] = mlb.transform(job_skills_map['skill_id']).tolist()

# Merge candidate and job data along with encoded skills
data = []
for _, candidate in candidates_df.iterrows():
    candidate_skills = candidate_skills_map[candidate_skills_map['candidate_id'] == candidate['candidate_id']]['skills_encoded'].values[0]
    for _, job in jobs_df.iterrows():
        job_skills = job_skills_map[job_skills_map['job_id'] == job['job_id']]['skills_encoded'].values[0]
        
        # Calculate match based on common skills
        common_skills = np.sum(np.minimum(candidate_skills, job_skills))
        match_percentage = common_skills / np.sum(job_skills) * 100 if np.sum(job_skills) > 0 else 0
        
        # Combine candidate and job features
        data.append({
            'desired_salary': candidate['desired_salary'],
            'job_type': job['job_type'],
            'location': candidate['location'],
            'candidate_skills': candidate_skills,
            'job_skills': job_skills,
            'match_percentage': match_percentage
        })

# Create a DataFrame from the data
match_df = pd.DataFrame(data)

# Convert skills from lists to columns
candidate_skills_df = pd.DataFrame(match_df['candidate_skills'].tolist(), index=match_df.index)
candidate_skills_df.columns = [f'candidate_skill_{i}' for i in range(candidate_skills_df.shape[1])]

job_skills_df = pd.DataFrame(match_df['job_skills'].tolist(), index=match_df.index)
job_skills_df.columns = [f'job_skill_{i}' for i in range(job_skills_df.shape[1])]

# Concatenate skills with the main DataFrame
match_df.reset_index(drop=True, inplace=True)
match_df = pd.concat([match_df, candidate_skills_df, job_skills_df], axis=1)
match_df.drop(['candidate_skills', 'job_skills'], axis=1, inplace=True)

# Step 2: Build the Neural Network

# Define features and target
X = match_df.drop('match_percentage', axis=1)
y = match_df['match_percentage']

# Update feature lists
categorical_features = ['job_type', 'location']
numerical_features = ['desired_salary'] + [col for col in match_df.columns if 'skill_' in col]

# Update preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse=False), categorical_features)
    ])

# Fit the preprocessor to get the number of features
X_preprocessed = preprocessor.fit_transform(X)
n_features = X_preprocessed.shape[1]

# Update model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('nn', keras.Sequential([
        layers.Input(shape=(n_features,)),  # Specify input shape
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer for match percentage
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
